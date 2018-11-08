/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "common_coll.h"
#include "enqueue.h"
#include "collectives.h"

ncclResult_t ncclAllReduceFunc(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  size_t nbytes = count*ncclTypeSize(datatype);
  fprintf(stderr, "Before opcount\n");
  INFO(COLL,"opCount %lx sendbuff %p recvbuff %p count %zi size %zi datatype %d op %d comm %p [nranks=%d] stream %p", comm->opCount, sendbuff, recvbuff, count, nbytes, datatype, op, comm, comm->nRanks, stream);
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, nbytes, cudaMemcpyDeviceToDevice, stream));
  } else {
    fprintf(stderr, "befor save proxies\n");
    NCCLCHECK(transportSaveProxies(ALLREDUCE_SUBSTEPS, ALLREDUCE_BUFCHUNKS, (comm->nRanks)*2-2, comm->nRanks, nbytes, proxyPatternRing, comm));
    fprintf(stderr, "after save proxies\n");
    NCCLCHECK(saveKernel(ncclCollAllReduce, sendbuff, recvbuff, count, datatype, op, root, comm, stream, nbytes, comm->nRanks));
    //NCCLCHECK(transportSaveProxies(ALLREDUCE_SUBSTEPS, ALLREDUCE_BUFCHUNKS, (comm->nRanks)*2-2, comm->nRanks, nbytes, proxyPatternRing, comm));
    // NCCLCHECK(saveKernel(ncclCollAllReduce, sendbuff, recvbuff, count, datatype, op, root, comm, stream, nbytes, comm->nRanks));

  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  fprintf(stderr, "Enqueue check\n");
  return ncclEnqueueCheck(ncclAllReduceFunc, "AllReduce", sendbuff, recvbuff, count, datatype,
          op, 0, comm->nodeComm, stream);
}
