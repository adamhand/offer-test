package Offer.test_60;

import DataStructureAndAlgorithm.BiTree2.BiTreeNode;
import DataStructureAndAlgorithm.BiTree2.BiTreeUtils;

import java.util.LinkedList;
import java.util.Queue;

//把二叉树打印成多行
public class PrintMultiline {
    public static void printWithMultiline(BiTreeNode root){
        Queue<BiTreeNode> queue = new LinkedList<>();
        BiTreeNode temp;
        int nextLevel = 0;       //下一层需要打印的节点数
        int toBePrinted = 1;     //本层还需要打印的节点数
        if(root != null)
            queue.offer(root);
        while(!queue.isEmpty()){
            temp = queue.poll();
            System.out.print(temp.getValue()+" ");
            if(null != temp.getLeft()){
                nextLevel++;
                queue.offer(temp.getLeft());
            }
            if(null != temp.getRight()){
                nextLevel++;
                queue.offer(temp.getRight());
            }
            toBePrinted--;
            if(toBePrinted == 0){
                System.out.println();
                toBePrinted = nextLevel;
                nextLevel = 0;
            }
        }
    }

    public static void main(String[] args){
        BiTreeNode root = null;
        System.out.println("请输入二叉树：");
        root = BiTreeUtils.createBiTreePreOrder(root);

        printWithMultiline(root);
    }
}
