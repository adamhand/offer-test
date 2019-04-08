package Offer.test_39_1;

class TreeNode{
    int value;
    TreeNode left;
    TreeNode right;
}

//判断一棵二叉树是不是平衡二叉树
public class IsBalacned {
    private boolean isBalanced = true;

    public boolean IsBalanced_Solution(TreeNode root) {
        height(root);
        return isBalanced;
    }

    private int height(TreeNode root) {
        if (root == null || !isBalanced)
            return 0;
        int left = height(root.left);
        int right = height(root.right);
        if (Math.abs(left - right) > 1)
            isBalanced = false;
        return 1 + Math.max(left, right);
    }
}
