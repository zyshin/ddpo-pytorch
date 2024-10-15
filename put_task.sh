set -e

echo adding task to $TASK_NAME
$GPU_MANAGER_BIN put_task --session_name ${TASK_NAME} --cmd "/mnt/gemininjceph2/geminicephfs/wx-mm-spr-xxxx/yshzhu/repos/ddpo/ddpo-pytorch/run.sh"
# $GPU_MANAGER_BIN put_task --session_name ${TASK_NAME} --cmd "sleep infinity"
# $GPU_MANAGER_BIN active --session_name ${TASK_NAME}
echo done
