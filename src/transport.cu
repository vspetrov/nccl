/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "common_coll.h"
ncclResult_t doSharp (struct ncclProxyArgs* args);
extern struct ncclTransport p2pTransport;
extern struct ncclTransport shmTransport;
extern struct ncclTransport netTransport;

struct ncclTransport ncclTransports[NTRANSPORTS] = {
  p2pTransport,
  shmTransport,
  netTransport,
};

static void FifoPullArgs(struct transportProxyInfo* info, struct ncclProxyArgs *args, bool isSharp) {
  struct ncclProxyArgs *fifoArgs = info->argsFifo + (info->argsFifoHead % TRANSPORT_PROXY_FIFO_SIZE);
  if (isSharp) printf("%s:%d\n", __FUNCTION__, __LINE__);
  pthread_mutex_lock(&info->mutex);
  if (isSharp)  printf("%s:%d\n", __FUNCTION__, __LINE__);
  while (fifoArgs->active == 0) {
    pthread_cond_wait(&info->cond, &info->mutex);
    if (isSharp) printf("%s:%d\n", __FUNCTION__, __LINE__);
  }
  __sync_synchronize();
  memcpy(args, fifoArgs, sizeof(struct ncclProxyArgs));
  __sync_synchronize();
  fifoArgs->active = 0;
  pthread_cond_signal(&info->cond);
  if (isSharp) printf("%s:%d\n", __FUNCTION__, __LINE__);
  pthread_mutex_unlock(&info->mutex);
  info->argsFifoHead++;
}

static struct ncclProxyArgs* FifoGetNextArgs(struct transportProxyInfo* info) {
  if (info == NULL) return NULL;
  struct ncclProxyArgs* fifoArgs = info->argsFifo + (info->argsFifoTail % TRANSPORT_PROXY_FIFO_SIZE);
  pthread_mutex_lock(&info->mutex);
  while (fifoArgs->active == 1)
    pthread_cond_wait(&info->cond, &info->mutex);
  pthread_mutex_unlock(&info->mutex);
  info->argsFifoTail++;
  return fifoArgs;
}

static void FifoPushArgs(struct transportProxyInfo* info) {
  if (info == NULL) return;

  struct ncclProxyArgs* fifoArgs = info->argsFifo + ((info->argsFifoTail-1) % TRANSPORT_PROXY_FIFO_SIZE);
  if (fifoArgs->active == 0) return;

  pthread_mutex_lock(&info->mutex);
  pthread_cond_signal(&info->cond);
  pthread_mutex_unlock(&info->mutex);
}

static void WaitProxyReady(struct transportProxyInfo* info) {
  pthread_mutex_lock(&info->mutex);
  while (info->proxyReady == 0)
    pthread_cond_wait(&info->cond, &info->mutex);
  pthread_mutex_unlock(&info->mutex);
}

static void SetProxyReady(struct transportProxyInfo* info) {
  pthread_mutex_lock(&info->mutex);
  info->proxyReady = 1;
  pthread_cond_signal(&info->cond);
  pthread_mutex_unlock(&info->mutex);
}

static void StopProxy(struct transportProxyInfo* info) {
  struct ncclProxyArgs* fifoArgs = FifoGetNextArgs(info);
  fifoArgs->active = -1;
  FifoPushArgs(info);
}

#define RECV 0
#define SEND 1
#define SHARP 2

static bool NeedProxy(int type, int pattern, struct ncclRing* ring, int nranks) {
  enum proxyMode mode = proxyPatternMode(pattern);
  if (mode == proxyRing) return true;

  /* In chains, one rank does not need a proxy. Let's figure out which one it is */
  int root = proxyPatternRoot(pattern);
  // Which index in the reorganized rings should we compare root against */
  const int myrank = 0, nextrank = 1, prevrank = nranks-1;
  int index = mode == proxyFrom ?
      /*                            no recv /  no send    if root = */
      /* bcast  */ (type == RECV ?   myrank : nextrank ):
      /* reduce */ (type == RECV ? prevrank :   myrank );
  int rank = ring->userRanks[index];
  return (root != rank);
}

static void SaveProxy(struct ncclConnector* connector, struct ncclProxyArgs* args, int needProxy) {
  struct transportProxyInfo* info = connector->proxyInfo;
  if (info == NULL) return;
  struct ncclProxyArgs* fifoArgs = FifoGetNextArgs(info);
  args->needProxy = needProxy;
  __sync_synchronize();
  memcpy(fifoArgs, args, sizeof(struct ncclProxyArgs));
  __sync_synchronize();
  fifoArgs->active = 1;
}

ncclResult_t transportSaveProxies(int substeps, int subchunks, int nstepsPerRound, int nblocksPerRound, size_t nbytes, int pattern, struct ncclComm* comm) {
  int llMode, nrings, nthreads;
  ncclGetCollResource(comm, nbytes, &nrings, &nthreads, &llMode);
  nbytes       = llMode ? nbytes * 2    : nbytes;
  substeps     = llMode ? 1             : substeps;
  subchunks    = llMode ? NCCL_LL_CHUNKS : subchunks;
  int buffSize = llMode ? NCCL_LL_BUFF_SIZE : comm->rings[0].buffSize;

  int nrounds = (int)(DIVUP(nbytes, ((size_t)nrings * nblocksPerRound * (buffSize/subchunks)))); // Fixed 32-bit overflow
  int nsteps = nstepsPerRound * nrounds * substeps;
  TRACE(NET,"opCount %lx substeps %d subchunks %d nrounds %d nsteps %d comm %p", comm->opCount, subchunks, subchunks, nrounds, nsteps, comm);
  TRACE(NET,"opCount %lx nbytes %zi nrings %d buffSize %d pattern %d comm %p", comm->opCount, nbytes, nrings, buffSize, pattern, comm);
  for (int r=0; r<nrings; r++) {
    struct ncclRing* ring = comm->rings+((comm->myParams->gridDim.x+r)%comm->nRings);
    struct ncclProxyArgs args = { ring, substeps*subchunks, nsteps, comm->opCount, llMode, 0 };
    SaveProxy(&ring->recv, &args, NeedProxy(RECV, pattern, ring, comm->nRanks));
    SaveProxy(&ring->send, &args, NeedProxy(SEND, pattern, ring, comm->nRanks));
    SaveProxy(&ring->sharp, &args, 1);
  }
  return ncclSuccess;
}

ncclResult_t transportStartProxies(ncclComm* comm) {
  for (int r=0; r<comm->nRings; r++) {
    FifoPushArgs(comm->rings[r].send.proxyInfo);
    FifoPushArgs(comm->rings[r].recv.proxyInfo);
    FifoPushArgs(comm->rings[r].sharp.proxyInfo);
  }
  pthread_yield(); // Let other threads run
  return ncclSuccess;
}

void* persistentThread(void *opaqueInfo) {
  struct transportProxyInfo* info = (struct transportProxyInfo*)opaqueInfo;
  // We need to initialize the context before launching any NCCL cuda kernel,
  // otherwise we would create it during the first cudaMemcpyAsync inside the
  // proxy function and that would cause a deadlock
  cudaSetDevice(info->comm->cudaDev);
  // Signal the main thread the context is created and it can proceed.
  SetProxyReady(info);
  while (1) {
    struct ncclProxyArgs args;
    if (info->func == &doSharp) {
        fprintf(stderr, "BEFORE\n");
    }
    
    FifoPullArgs(info, &args, (info->func == &doSharp));
    if (info->func == &doSharp) {
        fprintf(stderr, "AFTER\n");
    }
    
    if (args.active == -1) {
      // Main thread asked to stop
      return NULL;
    }
    ncclResult_t res = info->func(&args);
    if (res != ncclSuccess) {
      WARN("%s:%d -> %d [Proxy thread error]", __FILE__, __LINE__, res);
    }
  }
}


ncclResult_t doSharp (struct ncclProxyArgs* args){
  fprintf(stderr,"waaaaaa\n");
  return ncclSuccess;
  
}

struct netSendResources {
  void* netSendComm;
  struct ncclSendMem* hostSendMem;
  struct ncclRecvMem* hostRecvMem;
  struct ncclSendMem* devHostSendMem;
  struct ncclRecvMem* devHostRecvMem;
  struct ncclSendMem* hostDevMem;
  int netDev;
  bool cudaSupport;
  struct ncclRecvMem* devNetMem;
  uint64_t llStep;
  uint64_t llLastCleaning;
};

ncclResult_t sharpProxy(struct ncclProxyArgs* args) {
  struct ncclRing* ring = args->ring;
  struct netSendResources* resources = (struct netSendResources*) (ring->sharp.transportResources);
  const int llMode = args->llMode;

  volatile uint64_t* prevTail = &resources->hostRecvMem->tail;

  struct ncclSendMem* prevMem = resources->hostDevMem ? resources->hostDevMem : resources->hostSendMem;
  volatile uint64_t* prevHead = llMode ? &prevMem->llHead : &prevMem->head;
  struct ncclRecvMem* localMem = resources->cudaSupport ? resources->devNetMem : resources->hostRecvMem;
  char* localBuff = llMode ? resources->hostRecvMem->llBuff : localMem->buff;
  volatile int* sizesFifo = llMode ? resources->hostRecvMem->llSizesFifo : resources->hostRecvMem->sizesFifo;
  int buffSize = llMode ? NCCL_LL_BUFF_SIZE : ring->buffSize;
  int sliceSize = buffSize / args->substeps;

  while (!(*prevHead)){ 
    ;;
  }
  printf("Gap1! sizesFifo = %d\n", sizesFifo[0]);
  ++(*prevTail);
  
  fprintf("sharpProxy\n");



  return ncclSuccess;
}

ncclResult_t sharpSetup(ncclTinfo_t* myOpaqueInfo, ncclTinfo_t* peerOpaqueInfo, struct ncclConnect* connectInfo, struct ncclRing* ring) {
  struct netSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  ring->sharp.transportResources = resources;

  struct netInfo* myInfo = (struct netInfo*)myOpaqueInfo;
  resources->cudaSupport = false;

  // Get user's GDR READ setting

  // Determine whether the GPU has NVLink
  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));

  resources->cudaSupport = true;

  int size = offsetof(struct ncclRecvMem, buff)+ring->buffSize;
  if (resources->cudaSupport) {
    NCCLCHECK(ncclCudaCalloc((char**)(&resources->devNetMem), size));
  }

  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostRecvMem, (void**)&resources->devHostRecvMem, size));
  NCCLCHECK(ncclCudaHostAlloc((void**)&resources->hostSendMem, (void**)&resources->devHostSendMem, size));

  return ncclSuccess;
}

ncclResult_t sharpConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  // Setup device pointers
  struct netSendResources* resources = (struct netSendResources*)send->transportResources;

  if (resources->cudaSupport) {
    send->conn.buff = resources->devNetMem->buff;
    // We don't use devMem for llMode because the CPU has to read the data
    send->conn.llBuff = resources->devHostRecvMem->llBuff;
  } else {
    send->conn.buff = resources->devHostRecvMem->buff;
    send->conn.llBuff = resources->devHostRecvMem->llBuff;
  }
  send->conn.tail = &resources->devHostRecvMem->tail;
  send->conn.opCount = &resources->devHostRecvMem->opCount;
  send->conn.fifo = resources->devHostRecvMem->sizesFifo;
  send->conn.llFifo = resources->devHostRecvMem->llSizesFifo;

  if (resources->hostDevMem == NULL) {
    send->conn.head = &resources->devHostSendMem->head;
    send->conn.llHead = &resources->devHostSendMem->llHead;
  }

  //Sharp comm init should be here.
  return ncclSuccess;
}

ncclResult_t transportCreateProxy(int type, struct ncclRing* ring, struct ncclComm* comm) {
  struct ncclConnector* connector;
  threadFunc_t proxyfunc;
  switch(type){
   case(SEND):
    connector = &ring->send;
    proxyfunc = (threadFunc_t) connector->transport->send.proxy;
    break; 
   case(RECV):
    connector = &ring->recv;
    proxyfunc = (threadFunc_t) connector->transport->recv.proxy;
    break;
   case(SHARP):
    connector = &ring->sharp;
    proxyfunc = (threadFunc_t) &sharpProxy;
    break;
   default:
    return ncclInvalidArgument;
  }

  if (proxyfunc) {
    TRACE(NET,"type %d ring %p proxyfunc %p comm %p", type, ring, proxyfunc, comm);
    struct transportProxyInfo* info;
    NCCLCHECK(ncclCalloc(&info, 1));
    connector->proxyInfo = info;
    info->comm = comm;
    info->cond = PTHREAD_COND_INITIALIZER;
    info->mutex = PTHREAD_MUTEX_INITIALIZER;
    info->func = proxyfunc;
    info->argsFifoHead = info->argsFifoTail = 0;
    info->proxyReady = 0;
    pthread_create(&connector->proxyInfo->thread, NULL, persistentThread, info);
    // Wait for thread to initialize its CUDA context.
    WaitProxyReady(info);
  }
  return ncclSuccess;
}

ncclResult_t transportDestroyProxy(struct ncclConnector* connector) {
  if (connector->proxyInfo) {
    StopProxy(connector->proxyInfo);
    pthread_join(connector->proxyInfo->thread, NULL);
    free(connector->proxyInfo);
    connector->proxyInfo = NULL;
  }
  return ncclSuccess;
}
