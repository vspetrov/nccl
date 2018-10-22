/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "ring.h"
#include "param.h"
#include "nvmlwrap.h"
#include "rings.h"
#include "bootstrap.h"
#include "transport.h"
#include "common_coll.h"
#include "group.h"
#include "utils.h"
#include "net.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sched.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

DebugLevel ncclDebugLevel;
uint64_t ncclDebugMask = INIT; // Default debug sub-system mask is INIT
pthread_mutex_t ncclDebugOutputLock;
FILE *ncclDebugFile = stdout;

#ifdef ENABLE_TRACE
std::chrono::high_resolution_clock::time_point ncclEpoch;
#endif

#if __CUDACC_VER_MAJOR__ >= 10 || (__CUDACC_VER_MAJOR__ >= 9 && __CUDACC_VER_MINOR__ >= 2)
#define NCCL_GROUP_CUDA_STREAM 0 // CGMD: CUDA 9.2,10.X Don't need to use an internal CUDA stream
#else
#define NCCL_GROUP_CUDA_STREAM 1 // CGMD: CUDA 9.0,9.1 Need to use an internal CUDA stream
#endif

NCCL_PARAM(GroupCudaStream, "GROUP_CUDA_STREAM", NCCL_GROUP_CUDA_STREAM);

NCCL_PARAM(CheckPointers, "CHECK_POINTERS", 0);

extern "C" __attribute__ ((visibility("default")))
ncclNet_t* ncclNet = NULL;

// We define this as weak to let tests redefine their own
#pragma weak ncclCudaCompCap
int ncclCudaCompCap() {
  int cudaDev;
  if (cudaGetDevice(&cudaDev) != cudaSuccess) return 0;
  int ccMajor;
  if (cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, cudaDev) != cudaSuccess) return 0;
  return ccMajor;
}
int ncclCudaFullCompCap() {
  int cudaDev;
  if (cudaGetDevice(&cudaDev) != cudaSuccess) return 0;
  int ccMajor, ccMinor;
  if (cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, cudaDev) != cudaSuccess) return 0;
  if (cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, cudaDev) != cudaSuccess) return 0;
  return ccMajor*10+ccMinor;
}

void initNet() {
  if (ncclNet != NULL) {
    INFO(INIT,"Using external Network %s", ncclNetName());
  } else {
    ncclNet = ncclIbSupport() ? &ncclNetIb : &ncclNetSocket;
    INFO(INIT,"Using internal Network %s", ncclNetName());
  }
}

NCCL_PARAM(LlThreshold, "LL_THRESHOLD", -2);
NCCL_PARAM(ThreadThreshold, "THREAD_THRESHOLD", NCCL_THREAD_THRESHOLD);

pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;
static bool initialized = false;
static ncclResult_t ncclInit() {
  if (initialized) return ncclSuccess;
  pthread_mutex_lock(&initLock);
  if (!initialized) {
    initEnv();
    initDebug();
    initNet();
    initialized = true;
  }
  pthread_mutex_unlock(&initLock);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetVersion, int* version);
ncclResult_t ncclGetVersion(int* version) {
  if (version == NULL) return ncclInvalidArgument;
  *version = NCCL_VERSION_CODE;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  NCCLCHECK(ncclInit());
  NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));
  return bootstrapGetUniqueId(out);
}

static ncclResult_t commFree(ncclComm_t comm) {
  if (comm == NULL)
    return ncclSuccess;

  CUDACHECK(cudaFree(comm->devComm));

  for (int ring=0; ring<comm->nRings; ring++)
    NCCLCHECK(freeRing(comm->rings+ring));

  if (comm->doneEvent != NULL)
    CUDACHECK(cudaEventDestroy(comm->doneEvent));

  if (comm->launchMode == ncclComm::GROUP) {
    CUDACHECK(cudaStreamDestroy(comm->groupStream));
  }

  // Last rank frees shared resources between threads
  int isLast;
  NCCLCHECK(ncclCpuBarrierIn(comm, &isLast));
  if (isLast) {
    free(comm->intraBarrier);
    free(comm->intraParams);
    free(comm->intraCudaDevs);
    free(comm->intraCGMode);
    free(comm->intraCC);
  }

  free(comm);
  return ncclSuccess;
}

static ncclResult_t commAlloc(ncclComm_t* comret, int ndev, int rank) {
  if (ndev < 1) {
    WARN("invalid device count (%d) requested", ndev);
    return ncclInvalidArgument;
  }
  if (rank >= ndev || rank < 0) {
    WARN("rank %d exceeds ndev=%d", rank, ndev);
    return ncclInvalidArgument;
  }

  // Try to create a CUDA object right away. If there is something wrong with
  // the device we're on (failure cause #1) , better know it early.
  cudaEvent_t doneEvent;
  CUDACHECK(cudaEventCreateWithFlags(&doneEvent, cudaEventDisableTiming));

  struct ncclComm* comm;
  NCCLCHECK(ncclCalloc(&comm, 1));

  INFO(INIT,"comm %p rank %d nranks %d", comm, rank, ndev);
  comm->rank = rank;
  comm->nRanks = ndev;
  cudaGetDevice(&comm->cudaDev);
  comm->doneEvent = doneEvent;
  comm->llThreshold = ncclParamLlThreshold();
  comm->threadThreshold = ncclParamThreadThreshold();
  comm->checkPointers = ncclParamCheckPointers() == 1 ? true : false;
#if __CUDACC_VER_MAJOR__ >= 10 || (__CUDACC_VER_MAJOR__ >= 9 && __CUDACC_VER_MINOR__ >= 2)
  comm->groupCudaStream = ncclParamGroupCudaStream();
#else
  // Don't allow the user to overload the default setting in older CUDA builds
  comm->groupCudaStream = NCCL_GROUP_CUDA_STREAM;
#endif

  comm->argsptr = &comm->args;

  *comret = comm;
  return ncclSuccess;
}

static ncclResult_t devCommSetup(ncclComm_t comm) {
  // Fully duplicate the comm on the device
  NCCLCHECK(ncclCudaCalloc(&comm->devComm, 1));
  // Copy the comm on the device
  NCCLCHECK(ncclCudaMemcpy(comm->devComm, comm, 1));
  // Copy userRanks
  for (int r=0; r<comm->nRings; r++) {
    NCCLCHECK(ncclCudaMemcpy(comm->rings[r].devUserRanks, comm->rings[r].userRanks, comm->nRanks));
  }
  return ncclSuccess;
}

// Pre-process the string so that running "strings" on the lib can quickly reveal the version.
#define STR2(v) #v
#define STR(v) STR2(v)
#define VERSION_STRING "NCCL version " STR(NCCL_MAJOR) "." STR(NCCL_MINOR) "." STR(NCCL_PATCH) NCCL_SUFFIX "+cuda" STR(CUDA_MAJOR) "." STR(CUDA_MINOR)
static void showVersion() {
  static int shown = 0;
  if (shown == 0 && ncclDebugLevel >= VERSION) {
    printf("%s\n", VERSION_STRING);
    fflush(stdout);
    if (ncclDebugFile != stdout)
      INFO(ALL,"%s", VERSION_STRING); // Also log NCCL version in one of the files
    shown = 1;
  }
}

static ncclResult_t fillInfo(struct ncclInfo* info, int rank) {
  for (int t=0; t<NTRANSPORTS; t++) {
    NCCLCHECK(ncclTransports[t].fillInfo(info->tinfo+t, rank));
  }
  return ncclSuccess;
}

template <int type>
static ncclResult_t selectTransport(struct ncclInfo* myInfo, struct ncclInfo* peerInfo, struct ncclConnect* connect, struct ncclTransport** transportRet, struct ncclRing* ring) {
  for (int t=0; t<NTRANSPORTS; t++) {
    struct ncclTransport *transport = ncclTransports+t;
    struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    ncclTvalue_t ret = 0;
    NCCLCHECK(transport->canConnect(&ret, myInfo->tinfo+t, peerInfo->tinfo+t));
    if (ret > 0) {
      NCCLCHECK(transportComm->setup(myInfo->tinfo+t, peerInfo->tinfo+t, connect, ring));
      *transportRet = transport;
      return ncclSuccess;
    }
  }
  WARN("No transport found !");
  *transportRet = NULL;
  return ncclInternalError;
}

static ncclResult_t setupRing(struct ncclComm* comm, int ringid, int rank, int nranks, int* ringRanks, struct ncclInfo* allInfo, struct ncclConnect* connect) {
  NCCLCHECK(initRing(comm, ringid));

  struct ncclRing* ring = comm->rings+ringid;
  // Reorganize ranks to start with rank.
  int shift;
  for (shift = 0; shift<nranks; shift++) {
    if (ringRanks[shift] == rank) {
      break;
    }
  }
  for (int i=0; i<nranks; i++) {
    ring->userRanks[i] = ringRanks[(i+shift)%nranks];
  }
  int prev = ring->userRanks[nranks-1];
  int next = ring->userRanks[1];

  NCCLCHECK(selectTransport<0>(allInfo+rank, allInfo+prev, connect+0, &ring->recv.transport, ring));
  NCCLCHECK(selectTransport<1>(allInfo+rank, allInfo+next, connect+1, &ring->send.transport, ring));
  NCCLCHECK(transportCreateProxy(0, ring, comm));
  NCCLCHECK(transportCreateProxy(1, ring, comm));
  return ncclSuccess;
}


static ncclResult_t setupSharp(struct ncclComm* comm, int ringid, int rank, int nranks, int* ringRanks, struct ncclInfo* allInfo, struct ncclConnect* connect) {
  NCCLCHECK(initRing(comm, ringid));

  struct ncclRing* ring = comm->rings+ringid;
  // Reorganize ranks to start with rank.
  int shift;
  for (shift = 0; shift<nranks; shift++) {
    if (ringRanks[shift] == rank) {
      break;
    }
  }
  for (int i=0; i<nranks; i++) {
    ring->userRanks[i] = ringRanks[(i+shift)%nranks];
  }
  int prev = ring->userRanks[nranks-1];
  int next = ring->userRanks[1];

  NCCLCHECK(selectTransport<0>(allInfo+rank, allInfo+prev, connect+0, &ring->recv.transport, ring));
  NCCLCHECK(selectTransport<1>(allInfo+rank, allInfo+next, connect+1, &ring->send.transport, ring));
  NCCLCHECK(transportCreateProxy(0, ring, comm));
  NCCLCHECK(transportCreateProxy(1, ring, comm));
  return ncclSuccess;
}




static ncclResult_t fillConnect(struct ncclInfo* allInfo, int nranks, int rank, int* connectTransport, ncclTvalue_t* connectValue) {
  for (int r=0; r<nranks; r++) {
    connectTransport[r] = -1;
    for (int t=0; t<NTRANSPORTS; t++) {
      NCCLCHECK(ncclTransports[t].canConnect(connectValue+r, allInfo[rank].tinfo+t, allInfo[r].tinfo+t));
      if (connectValue[r] > 0) {
        connectTransport[r] = t;
        break;
      }
    }
  }
  return ncclSuccess;
}

static void swap(void* mem1, void* mem2, int size) {
  char tmp[size];
  memcpy(tmp, mem1, size); memcpy(mem1, mem2, size); memcpy(mem2, tmp, size);
}

#define MAXWIDTH 20
#define PREFIXLEN 15
#define STRLENGTH (PREFIXLEN+4*MAXWIDTH)
void dumpMatrix(int* connectMatrix, int nranks) {
  char line[STRLENGTH+1];
  line[STRLENGTH] = '\0';
  memset(line, ' ', STRLENGTH);
  for (int j=0; j<nranks && j<MAXWIDTH; j++) sprintf(4+line+4*j, " %3d", j);
  INFO(INIT,"%s", line);
  for (int i=0; i<nranks; i++) {
    memset(line, ' ', STRLENGTH);
    sprintf(line, "%3d ", i);
    for (int j=0; j<nranks && j<MAXWIDTH; j++) sprintf(4+line+4*j, " %3d", connectMatrix[i*nranks+j]);
    INFO(INIT,"%s", line);
  }
}

void dumpLine(int* values, int nranks, const char* prefix) {
  int prefixlen = strlen(prefix);
  char line[STRLENGTH+1];
  line[STRLENGTH] = '\0';
  memset(line, ' ', STRLENGTH);
  strncpy(line, prefix, PREFIXLEN);
  for (int i=0; i<nranks && i<MAXWIDTH; i++) sprintf(line+prefixlen+4*i, " %3d", values[i]);
  INFO(INIT,"%s", line);
}

static ncclResult_t buildRings(int nrings, int* rings, int rank, int nranks, int* prev, int* next) {
  for (int r=0; r<nrings; r++) {
    char prefix[30];
    /*sprintf(prefix, "[%d] Ring %d Prev : ", rank, r);
    dumpLine(prev+r*nranks, nranks, prefix);
    sprintf(prefix, "[%d] Ring %d Next : ", rank, r);
    dumpLine(next+r*nranks, nranks, prefix);*/

    int current = rank;
    for (int i=0; i<nranks; i++) {
      rings[r*nranks+i] = current;
      current = next[r*nranks+current];
    }
    sprintf(prefix, "Ring %02d : ", r);
    if (rank == 0) dumpLine(rings+r*nranks, nranks, prefix);
    if (current != rank) {
      WARN("Error : ring %d does not loop back to start (%d != %d)", r, current, rank);
      return ncclInternalError;
    }
    // Check that all ranks are there
    for (int i=0; i<nranks; i++) {
      int found = 0;
      for (int j=0; j<nranks; j++) {
        if (rings[r*nranks+j] == i) {
          found = 1;
          break;
        }
      }
      if (found == 0) {
        WARN("Error : ring %d does not contain rank %d", r, i);
        return ncclInternalError;
      }
    }
  }
  return ncclSuccess;
}

void* waitForNonNullPtr(void* p) {
  volatile void** ptr = (volatile void**) p;
  while (*ptr == NULL) sched_yield();
  return (void*)*ptr;
}

ncclResult_t initParams(struct ncclComm* comm) {
  struct cudaLaunchParams* params = comm->myParams = comm->intraParams+comm->intraRank;
  params->args = &comm->argsptr;
  params->stream = NULL;
  params->sharedMem = 0;
  params->blockDim.x = 0; params->blockDim.y = params->blockDim.z = 1;
  params->gridDim.x = 0; params->gridDim.y = params->gridDim.z = 1;
  return ncclSuccess;
}

// Allocate/Set Intra Process Structures and set CG options
ncclResult_t ncclCommSetIntra(struct ncclComm* comm, int rank, int ranks, struct ncclComm* comm0) {
  comm->intraRank = rank;
  comm->intraRanks = ranks;
  comm->intraPhase = 0;

  // Alloc shared structures
  if (rank == 0) {
    assert(comm == comm0);
    int* bar;
    NCCLCHECK(ncclCalloc(&bar, 2));
    bar[0] = bar[1] = 0;
    comm->intraBarrier = bar;
    NCCLCHECK(ncclCalloc(&comm->intraParams, comm->intraRanks));
    NCCLCHECK(ncclCalloc(&comm->intraCudaDevs, comm->intraRanks));
    int* CGMode;
    NCCLCHECK(ncclCalloc(&CGMode, 1));
    *CGMode = 0x11;
    comm->intraCGMode = CGMode;
    int* CC;
    NCCLCHECK(ncclCalloc(&CC, 1));
    *CC = ncclCudaFullCompCap();
    comm->intraCC = CC;
  } else {
    comm->intraBarrier = (int*)waitForNonNullPtr(&comm0->intraBarrier);
    comm->intraParams = (struct cudaLaunchParams*)waitForNonNullPtr(&comm0->intraParams);
    comm->intraCudaDevs = (int*)waitForNonNullPtr(&comm0->intraCudaDevs);
    comm->intraCGMode = (int*)waitForNonNullPtr(&comm0->intraCGMode);
    comm->intraCC = (int*)waitForNonNullPtr(&comm0->intraCC);
  }
  comm->intraCudaDevs[comm->intraRank] = comm->cudaDev;
  NCCLCHECK(initParams(comm));

  int cgMdLaunch = 0;

  // Set CG Mode
  comm->launchMode = ncclComm::GROUP;
  char* str = getenv("NCCL_LAUNCH_MODE");
  if (comm->intraRanks == 1 || (str && strcmp(str, "PARALLEL") == 0)) {
    comm->launchMode = ncclComm::PARALLEL;
  }
  if (comm->launchMode == ncclComm::GROUP) {
    CUDACHECK(cudaStreamCreateWithFlags(&comm->groupStream, cudaStreamNonBlocking));
#if __CUDACC_VER_MAJOR__ >= 9
    if (*comm->intraCC && (ncclCudaFullCompCap() == *comm->intraCC)) {
      // Check whether the GPU supports Cooperative Group Multi Device Launch
      (void) cudaDeviceGetAttribute(&cgMdLaunch, cudaDevAttrCooperativeMultiDeviceLaunch, comm->cudaDev);
    }
#endif
  }

  // Disable cgMdLaunch if any rank does not support it
  if (cgMdLaunch == 0) {
    *comm->intraCGMode = 0x10;
  }
  return ncclSuccess;
}

#if 0
static ncclResult_t sharpCommAlloc(struct ncclComm* comm, void *commState) {
    int nccl_sharp_np = -1;
    char *var = getenv("NCCL_SHARP_NP");
    if (var) {
        nccl_sharp_np = atoi(var);
    }
    int nnodes = comm->nRanks;
    if (nccl_sharp_np != -1 && nnodes >= nccl_sharp_np) {
        /*Initialize sharp ctx*/
        char *hostlist = (char*)malloc(max_hostlen*nnodes);
        //int *ranks_map = (int*)malloc(sizeof(int)*nnodes);
        int *ranks_map;
        CUDACHECK(cudaMallocManaged(&ranks_map, sizeof(int)*nnodes));
        ranks_map[netLocalRank] = myrank;
        NCCLCHECK(ncclAllGather(&ranks_map[netLocalRank], ranks_map, 1, ncclInt, (*newcomm)->netComm, s));
        CUDACHECK(cudaStreamSynchronize(s));
        fprintf(stderr,"RANKS: %d, %d\n", ranks_map[0], ranks_map[1]);
        hostlist[0]='\0';
        for (i=0;i<nnodes;i++) {
            strcat(hostlist,hosts + max_hostlen*ranks_map[i]);
            if (i < nnodes-1) {
                strcat(hostlist,",");
            }
        }
        fprintf(stderr,"HOSTS: %s\n", hostlist);
        cudaFree(ranks_map);
        static struct sharp_coll_context *sharpCtx = NULL;
        struct sharp_coll_init_spec init_spec = {0};
        init_spec.progress_func  = NULL;
        init_spec.job_id = 0xdeadbeef;
        init_spec.hostlist = hostlist;
        init_spec.world_rank = netLocalRank;
        init_spec.world_size = nnodes;;
#if SHARP_API > SHARP_VERSION(1,4)
        init_spec.world_local_rank = netLocalRank;
        init_spec.enable_thread_support = 0;
#endif
        init_spec.group_channel_idx = 0; //TODO support Yaniv's sharp comm layout
        init_spec.oob_colls.barrier = oob_barrier;
        init_spec.oob_colls.bcast = oob_bcast;
        init_spec.oob_colls.gather = oob_gather;
        init_spec.config = sharp_coll_default_config;
        init_spec.config.user_progress_num_polls = 10;
        char *dev = getenv("NCCL_SHARP_DEV");
        init_spec.config.ib_dev_list = dev ? dev : "mlx5_0:1";

        if (sharp_coll_init(&init_spec, &sharpCtx) < 0) {
            fprintf(stderr, "SHARP COLL INIT ERROR\n");
            return ncclInternalError;
        } else {
            fprintf(stderr, "SHARP INIT SUCCESS\n");
        }
#if 0
        //TODO need to move it down
        /* Initialize sharp communicator */

        struct sharp_coll_comm_init_spec comm_spec;
        uint32_t *gwr = NULL;
        comm_spec.rank      = netLocalRank;
        comm_spec.size      = nnodes;

#if SHARP_API > SHARP_VERSION(1,4)
        gwr = (uint32_t*)malloc(nnodes*sizeof(uint32_t));
        for (i=0; i<nnodes; i++) {
            gwr[i] = i;
        }
        comm_spec.group_world_ranks = gwr;
#endif
        comm_spec.is_comm_world = 1;
        comm_spec.oob_ctx   = (void*)(*newcomm)->netComm;

        int ret = sharp_coll_comm_init(sharpCtx, &comm_spec,
                                       (struct sharp_coll_comm **)&(*newcomm)->sharpComm);
        if (gwr) free(gwr);
        if (ret < 0) {
            if (myrank == 0)
                fprintf(stderr, "sharp group create failed:%s(%d)\n", sharp_coll_strerror(ret), ret);
            return ncclInternalError;
        }
#endif
    }
    return ncclSuccess;
}
#endif
static ncclResult_t initTransportsRank(struct ncclComm* comm, ncclUniqueId* commId, bool initSharp, bool initRing) {
  int rank = comm->rank;
  int nranks = comm->nRanks;
  void* commState;
  NCCLCHECK(bootstrapInit(commId, rank, nranks, &commState));

  struct ncclInfo* allInfo;
  NCCLCHECK(ncclCalloc(&allInfo, nranks));
  NCCLCHECK(fillInfo(allInfo+rank, rank));
  NCCLCHECK(bootstrapAllGather(commState, allInfo, sizeof(struct ncclInfo)));

  int* connectTransport;
  ncclTvalue_t* connectValue;
  NCCLCHECK(ncclCalloc(&connectTransport, nranks*nranks));
  NCCLCHECK(ncclCalloc(&connectValue, nranks*nranks));

  NCCLCHECK(fillConnect(allInfo, nranks, rank, connectTransport+nranks*rank, connectValue+nranks*rank));
  NCCLCHECK(bootstrapAllGather(commState, connectTransport, nranks*(sizeof(int))));
  NCCLCHECK(bootstrapAllGather(commState, connectValue, nranks*(sizeof(ncclTvalue_t))));
  //if (rank == 0) dumpMatrix(connectTransport, nranks);
  //if (rank == 0) dumpMatrix(connectValue, nranks);

  // Get my rings
  int nrings;
  int* prev, *next;
  NCCLCHECK(ncclCalloc(&prev, nranks*MAXRINGS));
  NCCLCHECK(ncclCalloc(&next, nranks*MAXRINGS));
  comm->nThreads = getDefaultThreads();
  NCCLCHECK(ncclGetRings(&nrings, &comm->nThreads, rank, nranks, connectTransport, connectValue, prev, next));
  free(connectTransport);
  free(connectValue);

  // Find max nThreads
  int allData[nranks];
  allData[rank] = comm->nThreads;
  NCCLCHECK(bootstrapAllGather(commState, allData, sizeof(int)));
  for (int i=0; i<nranks; i++)
    comm->nThreads = std::max(allData[i], comm->nThreads);
  if (rank == 0) INFO(INIT,"Using %d threads", comm->nThreads);

  // Determine the minimum CUDA Compute capability of all GPUs
  int myCompCap = ncclCudaCompCap();
  int minCompCap = myCompCap;
  allData[rank] = myCompCap;
  NCCLCHECK(bootstrapAllGather(commState, allData, sizeof(int)));
  for (int i=0; i<nranks; i++)
    minCompCap = std::min(allData[i], minCompCap);
  if (rank == 0) INFO(INIT,"Min Comp Cap %d", minCompCap);

  // Find min nrings across ranks
  allData[rank] = nrings;
  NCCLCHECK(bootstrapAllGather(commState, allData, sizeof(int)));
  for (int i=0; i<nranks; i++)
    nrings = std::min(allData[i], nrings);

  // Exchange data with others to build complete rings
  comm->nRings = nrings;
  for (int r=0; r<nrings; r++) {
    NCCLCHECK(bootstrapAllGather(commState, prev+r*nranks, sizeof(int)));
    NCCLCHECK(bootstrapAllGather(commState, next+r*nranks, sizeof(int)));
  }
  int *rings;
  NCCLCHECK(ncclCalloc(&rings, nranks*MAXRINGS));
  NCCLCHECK(buildRings(nrings, rings, rank, nranks, prev, next));
  free(prev);
  free(next);

  // Connect with prev/next for each ring
  for (int r=0; r<nrings; r++) {
    int* ringRanks = rings+r*nranks;
    struct ncclRing *ring = comm->rings+r;
    struct ncclConnect connect[2];
    NCCLCHECK(setupRing(comm, r, rank, nranks, ringRanks, allInfo, connect));
    NCCLCHECK(bootstrapRingExchange(commState, connect, ring->userRanks[nranks-1], ring->userRanks[1], sizeof(struct ncclConnect)));
    NCCLCHECK(ring->send.transport->send.connect(connect+1, &ring->send));
    NCCLCHECK(ring->recv.transport->recv.connect(connect+0, &ring->recv));
  }
  free(rings);
  free(allInfo);

  // Intra-process barrier setup
  struct rankInfo {
    uint64_t hostHash;
    uint64_t pidHash;
    struct ncclComm* comm;
  } rankInfos[nranks];
  rankInfos[rank].hostHash = getHostHash();
  rankInfos[rank].pidHash = getPidHash();
  rankInfos[rank].comm = comm;
  NCCLCHECK(bootstrapAllGather(commState, rankInfos, sizeof(struct rankInfo)));

  // Compute intra ranks
  int intraRank0 = -1, intraRank = -1, intraRanks = 0;
  for (int r=0; r<nranks; r++) {
    if ((rankInfos[r].hostHash == rankInfos[rank].hostHash) &&
        (rankInfos[r].pidHash == rankInfos[rank].pidHash)) {
      if (intraRanks == 0) intraRank0 = r;
      if (r == rank) intraRank = intraRanks;
      intraRanks++;
    }
  }
  TRACE(INIT,"hostHash[%d] %lx intraRank %d intraRanks %d intraRank0 %d",
      rank, rankInfos[rank].hostHash, intraRank, intraRanks, intraRank0);
  if (intraRank == -1 || intraRank0 == -1 || rankInfos[intraRank0].comm == NULL) {
    WARN("Failed to determine intra ranks hostHash[%d] %lx intraRank %d intraRanks %d intraRank0 %d",
        rank, rankInfos[rank].hostHash, intraRank, intraRanks, intraRank0);
    return ncclInternalError;
  }
  NCCLCHECK(ncclCommSetIntra(comm, intraRank, intraRanks, rankInfos[intraRank0].comm));

  // Barrier
  bootstrapClose(commState);
  return ncclSuccess;
}

bool SetCpuAffinity(int cudaDev, nvmlDevice_t* nvmlDevice) {
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  if (cudaDeviceGetPCIBusId(busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, cudaDev) != cudaSuccess) return false;
  if (wrapNvmlDeviceGetHandleByPciBusId(busId, nvmlDevice) != ncclSuccess) return false;
  if (wrapNvmlDeviceSetCpuAffinity(*nvmlDevice) != ncclSuccess) {
    WARN("Failed to set CPU affinity");
    return false;
  }
  return true;
}

ncclResult_t ncclCommInitRankSync(ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank, bool initSharp, bool initRing ) {
  cpu_set_t affinitySave;
  sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);

  NCCLCHECK(wrapNvmlSymbols());
  NCCLCHECK(wrapNvmlInit());

  // Make sure all host memory allocation are close to the GPU
  int cudaDev;
  nvmlDevice_t nvmlDevice;
  CUDACHECK(cudaGetDevice(&cudaDev));
  SetCpuAffinity(cudaDev, &nvmlDevice);
  ncclResult_t res;

  NCCLCHECKGOTO(commAlloc(newcomm, ndev, myrank), res, cleanup);
  NCCLCHECKGOTO(initTransportsRank(*newcomm, &commId , initSharp, initRing ), res, cleanup);
  NCCLCHECKGOTO(devCommSetup(*newcomm), res, cleanup);

  sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  NCCLCHECKGOTO(wrapNvmlShutdown(), res, cleanup);
  return ncclSuccess;
cleanup:
  *newcomm = NULL;
  sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  return res;
}

#include <inttypes.h>

int oob_barrier(void *ctx) {
    struct extState *st = (struct extState*)ctx;
    int nranks = st->nranks;
    void *tmp = malloc(nranks);
    bootstrapAllGather(st, tmp, 1);
    free(tmp);
    return 0;
}

int oob_gather(void *ctx, int root, void *sbuf, void *rbuf, int size) {
    struct extState *st = (struct extState*)ctx;
    int nranks = st->nranks;
    void *tmp = malloc(nranks*size);
    memcpy((void*)((ptrdiff_t)tmp + size*st->rank), sbuf, size);
    bootstrapAllGather(st, tmp, size);
    if (st->rank == root) {
        memcpy(rbuf, tmp, nranks*size);
    }
    free(tmp);
    return 0;
}

int oob_bcast(void *ctx, void *buf, int size, int root) {
    struct extState* state = (struct extState*)ctx;
    void *tmp = malloc(size*state->nranks);
    if (state->rank == root) {
        memcpy((void*)((ptrdiff_t)tmp+size*state->rank), buf, size);
    }
    bootstrapAllGather(state, tmp, size);
    if (state->rank != root) {
        memcpy(buf, (void*)((ptrdiff_t)tmp+size*root), size);
    }
    free(tmp);
    return 0;
}

NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int ndev, ncclUniqueId commId, int myrank);
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  char* env = getenv("NCCL_COMM_ID");
  if (env && myrank == 0) {
    NCCLCHECK(bootstrapCreateRoot(&commId, true));
  }

  NCCLCHECK(ncclInit());
  if (myrank == 0) showVersion();

  INFO(INIT,"rank %d nranks %d", myrank, nranks);

  // Make sure the CUDA runtime is initialized.
  CUDACHECK(cudaFree(NULL));

  NCCLCHECK(PtrCheck(newcomm, "CommInitRank", "newcomm"));
  if (nranks < 1 || myrank < 0 || myrank >= nranks) {
    WARN("Invalid rank requested : %d/%d", myrank, nranks);
    return ncclInvalidArgument;
  }

  if (ncclAsyncMode()) {
    int cudaDev;
    CUDACHECK(cudaGetDevice(&cudaDev));
    return ncclAsyncInit(ncclCommInitRankSync, cudaDev, newcomm, nranks, commId, myrank, false, true);
  } else {
      NCCLCHECK(ncclCommInitRankSync(newcomm, nranks, commId, myrank, false, true));
  }
  {
      int nccl_hier_coll_np = -1;

      char *var;
      var = getenv("NCCL_HIER_COLL_NP");
      if (var) {
          nccl_hier_coll_np = atoi(var);
      }
      if (nccl_hier_coll_np != -1 && nranks >= nccl_hier_coll_np) {
          char *hostname;
          CUDACHECK(cudaMallocManaged(&hostname, 256));
          gethostname(hostname, sizeof(hostname));
          int hostlen = strlen(hostname);
          cudaStream_t s;
          int *buf;
          CUDACHECK(cudaMallocManaged(&buf, 2*sizeof(int)));
          buf[0] = hostlen;
          CUDACHECK(cudaStreamCreate(&s));
          NCCLCHECK(ncclAllReduce(&buf[0], &buf[1], 1, ncclInt, ncclMax, *newcomm, s));
          CUDACHECK(cudaStreamSynchronize(s));
          int i;

          char *hosts;
          CUDACHECK(cudaMallocManaged(&hosts, buf[1]*sizeof(char)*nranks));
          int max_hostlen = buf[1]+1;
          NCCLCHECK(ncclAllGather(hostname, hosts, max_hostlen, ncclChar, *newcomm, s));
          CUDACHECK(cudaStreamSynchronize(s));
          int node_local_rank = -1;
          int local_ranks = 0;
          int node_leader_rank = -1;
          int is_single_node = 1;
          for (i=0; i<nranks; i++) {
              char *host = hosts + max_hostlen*i;
              if (0 == strcmp(hostname, host)) {
                  if (node_leader_rank == -1) {
                      node_leader_rank = i;
                  }
                  if (i == myrank) {
                      node_local_rank = local_ranks;
                  }
                  local_ranks++;
              } else {
                  is_single_node = 0;
              }
          }
          // fprintf(stderr, "node_local_rank %d, local_ranks %d, node_leader_rank %d\n",
          // node_local_rank, local_ranks, node_leader_rank);

          if (!is_single_node) {
              ncclUniqueId *uids;
              CUDACHECK(cudaMallocManaged(&uids, 2*sizeof(ncclUniqueId)*(nranks+1)));

              uids[0] = uids[1] = commId;
              if (myrank == node_leader_rank) {
                  NCCLCHECK(ncclGetUniqueId(&uids[0]));
                  fprintf(stderr, "UID: %" PRIx64 ":%" PRIx64 "\n", ((uint64_t*)&uids[0])[0], ((uint64_t*)&uids[0])[1]);
              }
              if (0 == myrank) {
                  NCCLCHECK(ncclGetUniqueId(&uids[1]));
                  fprintf(stderr, "UID: %" PRIx64 ":%" PRIx64 "\n", ((uint64_t*)&uids[1])[0], ((uint64_t*)&uids[1])[1]);
              }
              NCCLCHECK(ncclAllGather(uids, uids+2, 2*sizeof(ncclUniqueId), ncclChar,
                                      *newcomm, s));
              CUDACHECK(cudaStreamSynchronize(s));

              (*newcomm)->nodeComm = NULL;
              (*newcomm)->netComm  = NULL;
              (*newcomm)->sharpComm  = NULL;
              if (local_ranks > 1) {
                  ncclUniqueId node_comm_uid = (uids + 2 + node_leader_rank*2)[0];
                  fprintf(stderr,"NODE: rank %d, host %s, local_rank %d, local_size %d, node_leader %d, uid %" PRIx64 ":%" PRIx64 "\n",
                          myrank, hostname, node_local_rank, local_ranks, node_leader_rank, ((uint64_t*)&node_comm_uid)[0],
                          ((uint64_t*)&node_comm_uid)[1]);
                  NCCLCHECK(ncclCommInitRankSync(&((*newcomm)->nodeComm), local_ranks,
                                                 node_comm_uid, node_local_rank, true, false));
              }
              int netLocalRank = -1;
              if (node_leader_rank == myrank) {
                  ncclUniqueId net_comm_uid = (uids + 2)[1];
                  int nnodes = 0;
                  for (i=0; i<nranks; i++) {
                      if (!memcmp(&(uids+2+2*i)[0],&uids[0],sizeof(ncclUniqueId))) {
                          netLocalRank = nnodes;
                          if (local_ranks == 1) {
                              //TODO release node_comm_uid - no node level was created
                          }
                      }
                      if (memcmp(&(uids+2+2*i)[0], &commId, sizeof(ncclUniqueId))) {
                          nnodes++;
                      }
                  }
                  fprintf(stderr,"NET: rank %d, host %s, net_rank %d, net_size %d, uid %" PRIx64 ":%" PRIx64 "\n",
                          myrank, hostname, netLocalRank, nnodes, ((uint64_t*)&net_comm_uid)[0],
                          ((uint64_t*)&net_comm_uid)[1]);
                  NCCLCHECK(ncclCommInitRankSync(&((*newcomm)->netComm), nnodes,
                                                 net_comm_uid, netLocalRank, false, true));
              }
              CUDACHECK(cudaFree(uids));
          }
          CUDACHECK(cudaFree(buf));
          CUDACHECK(cudaFree(hosts));
          CUDACHECK(cudaFree(hostname));
          CUDACHECK(cudaStreamDestroy(s));
      }
  }
  return ncclSuccess;
}

static ncclResult_t initTransportsAll(struct ncclComm** comms, const int* devs, int nranks) {
  struct ncclInfo* allInfo;
  NCCLCHECK(ncclCalloc(&allInfo, nranks));
  for (int rank=0; rank<nranks; rank++) {
    CUDACHECK(cudaSetDevice(devs[rank]));
    NCCLCHECK(fillInfo(allInfo+rank, rank));
  }

  int* connectTransport;
  ncclTvalue_t* connectValue;
  NCCLCHECK(ncclCalloc(&connectTransport, nranks*nranks));
  NCCLCHECK(ncclCalloc(&connectValue, nranks*nranks));
  for (int rank=0; rank<nranks; rank++)
    NCCLCHECK(fillConnect(allInfo, nranks, rank, connectTransport+nranks*rank, connectValue+nranks*rank));

  int* prev, *prevFinal, *next, *nextFinal;
  NCCLCHECK(ncclCalloc(&prev, nranks*MAXRINGS));
  NCCLCHECK(ncclCalloc(&prevFinal, nranks*MAXRINGS));
  NCCLCHECK(ncclCalloc(&next, nranks*MAXRINGS));
  NCCLCHECK(ncclCalloc(&nextFinal, nranks*MAXRINGS));
  int nrings = MAXRINGS;
  int nthreads=0;
  int myCompCap = ncclCudaCompCap();
  int minCompCap = myCompCap;
  for (int rank=0; rank<nranks; rank++) {
    CUDACHECK(cudaSetDevice(devs[rank]));
    int nringsRank;
    int nthreadsRank = getDefaultThreads();
    myCompCap = ncclCudaCompCap();
    NCCLCHECK(ncclGetRings(&nringsRank, &nthreadsRank, rank, nranks, connectTransport, connectValue, prev, next));
    nrings = std::min(nrings, nringsRank);
    nthreads = std::max(nthreads, nthreadsRank);
    minCompCap = std::min(minCompCap, myCompCap);
    for (int ring=0; ring<nrings; ring++) {
      int index = ring*nranks+rank;
      prevFinal[index] = prev[index];
      nextFinal[index] = next[index];
    }
  }
  free(connectTransport);
  free(connectValue);
  free(prev);
  free(next);

  INFO(INIT,"Using %d threads", nthreads);
  INFO(INIT,"Min Comp Cap %d", minCompCap);

  int* rings;
  NCCLCHECK(ncclCalloc(&rings, nranks*MAXRINGS));
  NCCLCHECK(buildRings(nrings, rings, 0, nranks, prevFinal, nextFinal));
  free(prevFinal);
  free(nextFinal);

  for (int rank=0; rank<nranks; rank++) {
    comms[rank]->nRings = nrings;
    comms[rank]->nThreads = nthreads;
  }

  for (int r=0; r<nrings; r++) {
    struct ncclConnect connect[2*nranks];
    int* ringRanks = rings+r*nranks;
    for (int rank=0; rank<nranks; rank++) {
      CUDACHECK(cudaSetDevice(devs[rank]));
      NCCLCHECK(setupRing(comms[rank], r, rank, nranks, ringRanks, allInfo, connect+2*rank));
    }
    // RingExchange connect information
    for (int rank=0; rank<nranks; rank++) {
      // Swap rank->prev and prevRank->next
      struct ncclRing *ring = comms[rank]->rings+r;
      int prevRank = ring->userRanks[nranks-1];
      struct ncclConnect* prevRankNextConnect = connect+2*prevRank+1;
      struct ncclConnect* rankPrevConnect = connect+2*rank;
      swap(prevRankNextConnect, rankPrevConnect, sizeof(struct ncclConnect));
    }
    for (int rank=0; rank<nranks; rank++) {
      CUDACHECK(cudaSetDevice(devs[rank]));
      struct ncclRing *ring = comms[rank]->rings+r;
      NCCLCHECK(ring->send.transport->send.connect(connect+2*rank+1, &ring->send));
      NCCLCHECK(ring->recv.transport->recv.connect(connect+2*rank+0, &ring->recv));
    }
  }
  free(rings);
  free(allInfo);
  return ncclSuccess;
}


NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  NCCLCHECK(ncclInit());
  NCCLCHECK(wrapNvmlSymbols());
  NCCLCHECK(wrapNvmlInit());
  showVersion();

  INFO(INIT,"nranks %d", ndev);

  NCCLCHECK(PtrCheck(comms, "CommInitAll", "comms"));
  if (ndev < 1) {
    WARN("Invalid device count requested : %d", ndev);
    return ncclInvalidArgument;
  }

  ncclResult_t res;
  int savedDevice;
  int rank, cudaDev;
  ncclComm_t comm = NULL;
  nvmlDevice_t nvmlDevice;
  int ncclDevList[ndev];
  for (int i=0; i<ndev; i++) {
    ncclDevList[i] = devlist ? devlist[i] : i;
  }

  cudaGetDevice(&savedDevice);

  for(rank=0; rank<ndev; ++rank)
    comms[rank] = NULL;

  cpu_set_t affinitySave;
  sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);

  for (rank=0; rank<ndev; ++rank) {
    cudaDev = ncclDevList[rank];
    CUDACHECKGOTO(cudaSetDevice(cudaDev), res, cleanup);

    SetCpuAffinity(cudaDev, &nvmlDevice);

    NCCLCHECKGOTO(commAlloc(&comm, ndev, rank), res, cleanup);
    comms[rank] = comm;

    NCCLCHECKGOTO(ncclCommSetIntra(comm, rank, ndev, comms[0]), res, cleanup);
  }

  sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);

  NCCLCHECKGOTO(initTransportsAll(comms, ncclDevList, ndev), res, cleanup);

  for(rank=0; rank<ndev; ++rank) {
    cudaDev = ncclDevList[rank];
    CUDACHECKGOTO(cudaSetDevice(cudaDev), res, cleanup);
    NCCLCHECKGOTO(devCommSetup(comms[rank]), res, cleanup);
  }

  res = ncclSuccess;
  goto final;

cleanup:
  for(rank=0; rank<ndev; ++rank) {
    if(comms[rank] != NULL) {
      commFree(comms[rank]);
    }
  }

final:
  if(wrapNvmlShutdown() != ncclSuccess)
    INFO(INIT,"NCCL did not shutdown nvml properly");
  cudaSetDevice(savedDevice);
  sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  return res;
}

NCCL_API(ncclResult_t, ncclCommDestroy, ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm) {

  if (comm == NULL)
    return ncclSuccess;
  int savedDevice;
  CUDACHECK(cudaGetDevice(&savedDevice));
  int commDevice = comm->cudaDev;

  if (savedDevice != commDevice) {
    CUDACHECK(cudaSetDevice(commDevice));
  }

  NCCLCHECK(commFree(comm));

  if (savedDevice != commDevice)
    CUDACHECK(cudaSetDevice(savedDevice));

  return ncclSuccess;
}

NCCL_API(const char*, ncclGetErrorString, ncclResult_t code);
const char* ncclGetErrorString(ncclResult_t code) {
  switch (code) {
    case ncclSuccess                : return "no error";
    case ncclUnhandledCudaError     : return "unhandled cuda error";
    case ncclSystemError            : return "unhandled system error";
    case ncclInternalError          : return "internal error";
    case ncclInvalidArgument        : return "invalid argument";
    case ncclInvalidUsage           : return "invalid usage";
    default                         : return "unknown result code";
  }
}

NCCL_API(ncclResult_t, ncclCommCount, const ncclComm_t comm, int* count);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  NCCLCHECK(PtrCheck(comm, "CommCount", "comm"));
  NCCLCHECK(PtrCheck(count, "CommCount", "count"));
  *count = comm->nRanks;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCuDevice, const ncclComm_t comm, int* devid);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* devid) {
  NCCLCHECK(PtrCheck(comm, "CommCuDevice", "comm"));
  NCCLCHECK(PtrCheck(devid, "CommCuDevice", "devid"));
  *devid = comm->cudaDev;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommUserRank, const ncclComm_t comm, int* rank);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  NCCLCHECK(PtrCheck(comm, "CommUserRank", "comm"));
  NCCLCHECK(PtrCheck(rank, "CommUserRank", "rank"));
  *rank = comm->rank;
  return ncclSuccess;
}
