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
  INFO(COLL,"opCount %lx sendbuff %p recvbuff %p count %zi size %zi datatype %d op %d comm %p [nranks=%d] stream %p", comm->opCount, sendbuff, recvbuff, count, nbytes, datatype, op, comm, comm->nRanks, stream);
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff)
      CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, nbytes, cudaMemcpyDeviceToDevice, stream));
  } else {
    NCCLCHECK(transportSaveProxies(ALLREDUCE_SUBSTEPS, ALLREDUCE_BUFCHUNKS, (comm->nRanks)*2-2, comm->nRanks, nbytes, proxyPatternRing, comm));
    NCCLCHECK(saveKernel(ncclCollAllReduce, sendbuff, recvbuff, count, datatype, op, root, comm, stream, nbytes, comm->nRanks));
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);

extern ncclResult_t ncclReduceFunc(const void* sendbuff, void* recvbuff, const size_t count,
                                   ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
extern ncclResult_t ncclBroadcastFunc(const void* sendbuff, void* recvbuff, const size_t count,
                                      ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
    if (0 && (comm->nodeComm || comm->netComm)) {
        fprintf(stderr,"NodeCOMM Reduce\n");
        ncclResult_t ret = ncclSuccess;
        if (comm->nodeComm) {
            ret = ncclEnqueueCheck(ncclReduceFunc, "Reduce", sendbuff, recvbuff, count, datatype,
                                   op, 0, comm->nodeComm, stream);
        }
        cudaStreamSynchronize(stream);

        if (comm->netComm) {
            fprintf(stderr,"NET COMM ALLREDUCE\n");
            if (comm->sharpComm) {
                struct sharp_coll_reduce_spec reduce_spec;
                enum sharp_datatype sharp_type;
                enum sharp_reduce_op op_type;
                size_t dt_size;
                sharp_type = SHARP_DTYPE_FLOAT; //TODO map from ncclTYPE
                op_type = SHARP_OP_SUM; //TODO map from ncclOP

                dt_size = sizeof(float);//SHOULD be dtype dependent

                reduce_spec.sbuf_desc.buffer.ptr = recvbuff;

                void *mr = NULL;

                if (SHARP_COLL_SUCCESS != sharp_coll_reg_mr(comm->sharpCtx, recvbuff, count * dt_size, &mr)) {
                    fprintf(stderr, "SHARP REG MR FAILED\n");
                }
                reduce_spec.sbuf_desc.buffer.length = count * dt_size;
                reduce_spec.sbuf_desc.buffer.mem_handle = mr;
                reduce_spec.sbuf_desc.type = SHARP_DATA_BUFFER;
                reduce_spec.rbuf_desc.buffer.ptr = recvbuff;
                reduce_spec.rbuf_desc.buffer.length = count * dt_size;
                reduce_spec.rbuf_desc.buffer.mem_handle = mr;
                reduce_spec.rbuf_desc.type = SHARP_DATA_BUFFER;

                reduce_spec.sbuf_desc.mem_type = SHARP_MEM_TYPE_CUDA;
                reduce_spec.rbuf_desc.mem_type = SHARP_MEM_TYPE_CUDA;

                reduce_spec.length = count;
                reduce_spec.dtype = sharp_type;
                reduce_spec.op = op_type;

                if (SHARP_COLL_SUCCESS != sharp_coll_do_allreduce(comm->sharpComm, &reduce_spec)) {
                    fprintf(stderr, "SHARP ALLREDUCE FAILED\n");
                }

                if (SHARP_COLL_SUCCESS != sharp_coll_dereg_mr(comm->sharpCtx, mr)) {
                    fprintf(stderr, "SHARP DEREG MR FAILED\n");
                }

            } else {
                ret = ncclEnqueueCheck(ncclAllReduceFunc, "AllReduce", recvbuff, recvbuff, count, datatype,
                                       op, 0, comm->netComm, stream);
                cudaStreamSynchronize(stream);
            }
        }
        fprintf(stderr,"NODE COMM BCAST\n");
        if (comm->nodeComm) {
            ret = ncclEnqueueCheck(ncclBroadcastFunc, "Broadcast", recvbuff, recvbuff, count, datatype,
                                   ncclSum, 0, comm->nodeComm, stream);
        }
        return ret;
    } else {
        return ncclEnqueueCheck(ncclAllReduceFunc, "AllReduce", sendbuff, recvbuff, count, datatype,
                                op, 0, comm->nodeComm, stream);
    }
}
