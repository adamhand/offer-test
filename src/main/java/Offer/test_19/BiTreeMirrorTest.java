package Offer.test_19;

import DataStructureAndAlgorithm.BiTree.BiTree;
import DataStructureAndAlgorithm.BiTree2.BiTreeNode;
import DataStructureAndAlgorithm.BiTree2.BiTreeUtils;

import java.util.Stack;

public class BiTreeMirrorTest {
    public static void main(String[] args) {
        BiTreeNode root = null;
        System.out.println("请输入树：");
        root = BiTreeUtils.createBiTreePreOrder(root);

        BiTreeUtils.levelOrder(root);

        mirrorBiTreeRecursively(root);

        System.out.println();
        BiTreeUtils.levelOrder(root);

        System.out.println();
        mirrorBiTreeLevel(root);

        System.out.println();
        BiTreeUtils.levelOrder(root);
    }


    //递归
    public static void mirrorBiTreeRecursively(BiTreeNode root){
        if(root == null)
            return;
        if(root.getLeft() == null && root.getRight() == null)
            return;

        BiTreeNode node = root.getLeft();
        root.setLeft(root.getRight());
        root.setRight(node);

        if(root.getLeft() != null)
            mirrorBiTreeRecursively(root.getLeft());

        if(root.getRight() != null)
            mirrorBiTreeRecursively(root.getRight());
    }

    //迭代。层次遍历
    public static void mirrorBiTreeLevel(BiTreeNode root){
        if(root == null)
            return;
        if(root.getLeft() == null && root.getRight() == null)
            return;
        Stack<BiTreeNode> stack = new Stack<>();
        stack.push(root);
        while(!stack.isEmpty()){
            BiTreeNode node = stack.pop();
            swapSonNode(node);
            if(node.getRight() != null)
                stack.push(node.getRight());
            if(node.getLeft() != null)
                stack.push(node.getLeft());
        }
    }

    public static void swapSonNode(BiTreeNode root){
        BiTreeNode node = root.getLeft();
        root.setLeft(root.getRight());
        root.setRight(node);
    }
}
