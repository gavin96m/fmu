#!/bin/bash

# 设置分支名称
BRANCH_NAME="main"

# 检查当前分支是否干净（无修改）
check_local_changes() {
    if [ -z "$(git status --porcelain)" ]; then
        echo "No local changes to commit."
        return 1
    else
        echo "Local changes detected."
        return 0
    fi
}

# 检查远程是否有更新
check_remote_changes() {
    echo "Fetching latest changes from remote..."
    git fetch origin $BRANCH_NAME
    LOCAL=$(git rev-parse $BRANCH_NAME)
    REMOTE=$(git rev-parse origin/$BRANCH_NAME)
    BASE=$(git merge-base $BRANCH_NAME origin/$BRANCH_NAME)

    if [ "$LOCAL" = "$REMOTE" ]; then
        echo "No remote updates."
        return 1
    elif [ "$LOCAL" = "$BASE" ]; then
        echo "Remote changes detected. Need to pull."
        return 0
    else
        echo "Local and remote have diverged. Manual resolution may be required."
        return 2
    fi
}

# 拉取远程更新
pull_remote_changes() {
    echo "Pulling changes from remote..."
    git pull origin $BRANCH_NAME
}

# 提交本地更改
commit_local_changes() {
    echo "Staging local changes..."
    git add .
    echo "Enter a commit message:"
    read COMMIT_MESSAGE
    if [ -z "$COMMIT_MESSAGE" ]; then
        echo "Commit message cannot be empty. Aborting commit."
        return 1
    fi
    git commit -m "$COMMIT_MESSAGE"
}

# 推送本地更改
push_local_changes() {
    echo "Pushing changes to remote..."
    git push origin $BRANCH_NAME
}

# 主流程
echo "Checking repository status..."
check_remote_changes
REMOTE_STATUS=$?

if [ "$REMOTE_STATUS" -eq 0 ]; then
    pull_remote_changes
elif [ "$REMOTE_STATUS" -eq 2 ]; then
    echo "Local and remote histories have diverged. Please resolve conflicts manually."
    exit 1
fi

check_local_changes
LOCAL_STATUS=$?

if [ "$LOCAL_STATUS" -eq 0 ]; then
    commit_local_changes
    push_local_changes
else
    echo "No local changes to push."
fi

echo "Git workflow completed."

