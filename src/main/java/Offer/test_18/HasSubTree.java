package Offer.test_18;

import DataStructureAndAlgorithm.BiTree.BiTree;
import DataStructureAndAlgorithm.BiTree2.BiTreeNode;
import DataStructureAndAlgorithm.BiTree2.BiTreeUtils;

import java.util.Arrays;

//判断树B是不是树A的子树
public class HasSubTree {
    public static boolean HasSubtree(BiTreeNode root1, BiTreeNode root2) {
        if(root1 == null || root2 == null)
            return false;
        //要么当前结点已经是子树 要么当前结点的左孩子或右孩子存在子树
        return IsSubtree(root1,root2) || HasSubtree(root1.getLeft(),root2) || HasSubtree(root1.getRight(),root2);
    }

    public static boolean IsSubtree(BiTreeNode root1, BiTreeNode root2){
        if(root2 == null)
            return true;
        if(root1 == null)
            return false;
        if(root1.getValue() == root2.getValue())
            return IsSubtree(root1.getLeft(), root2.getLeft()) && IsSubtree(root1.getRight(), root2.getRight());
        else
            return false;
    }


    public static void main(String[] args) {
        BiTreeNode rootA = null;
        BiTreeNode rootB = null;

        System.out.println("请输入树A：");
        rootA = BiTreeUtils.createBiTreePreOrder(rootA);

        System.out.println("请输入树B：");
        rootB = BiTreeUtils.createBiTreePreOrder(rootB);

        System.out.println(HasSubtree(rootA, rootB));
    }
}
