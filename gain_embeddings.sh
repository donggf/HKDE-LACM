#!/bin/bash

#read -p "Please enter the value of k: " k_value
script_dir=$(pwd)

# 定义包含 Python 脚本的文件夹路径
SCRIPT_FOLDER=$script_dir  # 替换为你的脚本文件夹路径

# 定义 Python 文件名
PYTHON_SCRIPTS=("create_dir.py" "Split_dataset.py" "finetune_model.py" "gain_prediction.py" "Split_forward_and_reverse.py"  "Obtain_complete_embeddings.py" "Obtain_incorrect_complete_embeddings.py")

# 检查脚本文件夹是否存在
if [ ! -d "$SCRIPT_FOLDER" ]; then
    echo "Error: Script folder does not exist."
    exit 1
fi

# 遍历并运行每个 Python 脚本
for SCRIPT in "${PYTHON_SCRIPTS[@]}"; do
    echo -e "\n"
    echo "Running $SCRIPT..."
    python3 "$SCRIPT_FOLDER/$SCRIPT"
    if [ $? -ne 0 ]; then
        echo "Error: $SCRIPT failed to run."
        exit 1
    fi
done

#python predict_sh.py "$k_value"
echo "All scripts have been executed successfully."
