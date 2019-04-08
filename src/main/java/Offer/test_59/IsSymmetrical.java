package Offer.test_59;

import DataStructureAndAlgorithm.BiTree2.BiTreeNode;
import DataStructureAndAlgorithm.BiTree2.BiTreeUtils;

//判断一个二叉树是不是对称的
public class IsSymmetrical {
    public static boolean isSymmetrical(BiTreeNode root1, BiTreeNode root2){
        if(root1 == null && root2 == null)
            return true;
        if(root1 == null || root2 == null)
            return false;
        if(root1.getValue() != root2.getValue())
            return false;

        return isSymmetrical(root1.getLeft(), root2.getRight()) &&
               isSymmetrical(root1.getRight(), root2.getLeft());
    }

    public static boolean isSymmetrical(BiTreeNode root){
        return isSymmetrical(root.getLeft(), root.getRight());
    }

    public static void main(String[] args) {
        BiTreeNode root = null;
        System.out.println("请输入树：");
        root = BiTreeUtils.createBiTreePreOrder(root);

        System.out.println(isSymmetrical(root));
    }
}
