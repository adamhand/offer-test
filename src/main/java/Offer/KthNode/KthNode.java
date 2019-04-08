package Offer.KthNode;

class TreeNode{
    int value;
    TreeNode left;
    TreeNode right;
}

//二叉搜索树的第K大节点
public class KthNode {
    public static void main(String[] args) {


    }

    /**
     * 利用二叉搜索树中序遍历有序的特点
     */
    private TreeNode ret;
    private int cnt = 0;

    public TreeNode KthNode(TreeNode pRoot, int k) {
        inOrder(pRoot, k);
        return ret;
    }

    private void inOrder(TreeNode root, int k) {
        if (root == null || cnt >= k)
            return;
        inOrder(root.left, k);
        cnt++;
        if (cnt == k)
            ret = root;
        inOrder(root.right, k);
    }
}
