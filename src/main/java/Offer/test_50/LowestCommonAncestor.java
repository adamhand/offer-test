package Offer.test_50;

class TreeNode{
    int val;
    TreeNode left;
    TreeNode right;
}

//树中两个节点的最低公共祖先
public class LowestCommonAncestor {
    public static void main(String[] args) {

    }

    /**
     * 二叉查找树
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null)
            return root;
        if (root.val > p.val && root.val > q.val)
            return lowestCommonAncestor(root.left, p, q);
        if (root.val < p.val && root.val < q.val)
            return lowestCommonAncestor(root.right, p, q);
        return root;
    }

    /**
     * 普通二叉树
     */
//    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
//        if (root == null || root == p || root == q)
//            return root;
//        TreeNode left = lowestCommonAncestor(root.left, p, q);
//        TreeNode right = lowestCommonAncestor(root.right, p, q);
//        return left == null ? right : right == null ? left : root;
//    }
}
