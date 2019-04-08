package Offer.test_61;

import DataStructureAndAlgorithm.BiTree.BiTree;
import DataStructureAndAlgorithm.BiTree2.BiTreeNode;
import DataStructureAndAlgorithm.BiTree2.BiTreeUtils;

import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;

//按照 之 字形打印二叉树
public class PrintWithMultiline {
    //用栈实现
    public static void printMultiline(BiTreeNode root){
        if(root == null)
            return;
        Stack<BiTreeNode>[] levels = new Stack[2];
        //注意：如果没有下面三行的话，levels[current].push(root);会报空指针异常，
        //因为只是定义了一个存放Stack的容器，并没有对其进行初始化
        Stack<BiTreeNode> stack1 = new Stack<>();
        Stack<BiTreeNode> stack2 = new Stack<>();
        levels[0] = stack1;
        levels[1] = stack2;
        int current = 0;
        int next = 1;
        levels[current].push(root);
        while(!levels[0].isEmpty() || !levels[1].isEmpty()){
            BiTreeNode node = levels[current].peek();
            levels[current].pop();
            System.out.print(node.getValue()+" ");

            if(current == 0){
                if(node.getLeft() != null)
                    levels[next].push(node.getLeft());
                if(node.getRight() != null)
                    levels[next].push(node.getRight());
            }else {
                if(node.getRight() != null)
                    levels[next].push(node.getRight());
                if(node.getLeft() != null)
                    levels[next].push(node.getLeft());
            }

            if(levels[current].isEmpty()){
                System.out.println();
                current = 1 - current;
                next = 1 - next;
            }
        }
    }

    //用队列实现。未成功
    public static void printMultilineWithQueue(BiTreeNode root){
        if(root == null)
            return;
        Queue<BiTreeNode>[] levels = new Queue[2];
        Queue<BiTreeNode> queue1 = new LinkedList<>();
        Queue<BiTreeNode> queue2 = new LinkedList<>();
        levels[0] = queue1;
        levels[1] = queue2;
        int current = 0;
        int next = 1;
        boolean isOdd = true;      //是奇数层就不翻转；是偶数层就翻转

        levels[0].offer(root);
        while(!levels[0].isEmpty() || !levels[1].isEmpty()){
            BiTreeNode node = levels[current].poll();
            System.out.print(node.getValue()+" ");

            if(current == 0){
                if(node.getRight() != null)
                    levels[next].offer(node.getRight());
                if(node.getLeft() != null)
                    levels[next].offer(node.getLeft());
            }else {
                if(node.getLeft() != null)
                    levels[next].offer(node.getLeft());
                if(node.getRight() != null)
                    levels[next].offer(node.getRight());
            }

            if(levels[current].isEmpty()){
                System.out.println();
                current = 1 - current;
                next = 1 - next;
            }
        }
    }


    public static void main(String[] args) {
        BiTreeNode root = null;
        System.out.println("请输入二叉树：");
        root = BiTreeUtils.createBiTreePreOrder(root);

//        printMultiline(root);
        printMultilineWithQueue(root);
    }
}
