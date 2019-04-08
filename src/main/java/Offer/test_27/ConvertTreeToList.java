package Offer.test_27;

import DataStructureAndAlgorithm.BiTree.BiTree;
import DataStructureAndAlgorithm.BiTree2.BiTreeNode;
import DataStructureAndAlgorithm.BiTree2.BiTreeUtils;
import DataStructureAndAlgorithm.Tree.TreeNode;
import OuterClass.OuterClass_1.OuterClass;

import java.util.Stack;

public class ConvertTreeToList {
    public static void main(String[] args) {
        Test.test1();
        Test.test2();
        Test.test3();
        Test.test4();
        Test.test5();
    }

    /**
     * 方法一：非递归版
     * 解题思路：
     * 1.核心是中序遍历的非递归算法。
     * 2.修改当前遍历节点与前一遍历节点的指针指向。
     * @param root
     * @return
     */
    public static BiTreeNode ConvertBSTToBiList(BiTreeNode root) {
        if(root==null)
            return null;
        Stack<BiTreeNode> stack = new Stack<BiTreeNode>();
        BiTreeNode p = root;
        BiTreeNode pre = null;// 用于保存中序遍历序列的上一节点
        boolean isFirst = true;
        while(p != null || !stack.isEmpty()){
            while(p != null){
                stack.push(p);
                p = p.getLeft();
            }
            p = stack.pop();
            if(isFirst){
                root = p;// 将中序遍历序列中的第一个节点记为root
                pre = root;
                isFirst = false;
            }else{
                pre.setRight(p);
                p.setLeft(pre);
                pre = p;
            }
            p = p.getRight();
        }
        return root;
    }

    /**
     * 方法二：递归版
     * 解题思路：
     * 1.将左子树构造成双链表，并返回链表头节点。
     * 2.定位至左子树双链表最后一个节点。
     * 3.如果左子树链表不为空的话，将当前root追加到左子树链表。
     * 4.将右子树构造成双链表，并返回链表头节点。
     * 5.如果右子树链表不为空的话，将该链表追加到root节点之后。
     * 6.根据左子树链表是否为空确定返回的节点。
     * @param root
     * @return
     */

    public static BiTreeNode Convert(BiTreeNode root) {
        if(root==null)
            return null;
        if(root.getLeft()==null&&root.getRight()==null)
            return root;
        // 1.将左子树构造成双链表，并返回链表头节点
        BiTreeNode left = Convert(root.getLeft());
        BiTreeNode p = left;
        // 2.定位至左子树双链表最后一个节点
        while(p!=null&&p.getRight()!=null){
            p = p.getRight();
        }
        // 3.如果左子树链表不为空的话，将当前root追加到左子树链表
        if(left!=null){
            p.setRight(root);
            root.setLeft(p);
        }
        // 4.将右子树构造成双链表，并返回链表头节点
        BiTreeNode right = Convert(root.getRight());
        // 5.如果右子树链表不为空的话，将该链表追加到root节点之后
        if(right!=null){
            right.setLeft(root);
            root.setRight(right);
        }
        return left!=null?left:root;
    }


//    static BiTreeNode head = null;
//    static BiTreeNode realHead = null;
//    public static BiTreeNode Convert(BiTreeNode pRootOfTree) {
//        ConvertSub(pRootOfTree);
//        return realHead;
//    }
//
//    private static void ConvertSub(BiTreeNode pRootOfTree) {
//        if(pRootOfTree==null) return;
//        ConvertSub(pRootOfTree.getLeft());
//        if (head == null) {
//            head = pRootOfTree;
//            realHead = pRootOfTree;
//        } else {
//            head.setRight(pRootOfTree);
//            pRootOfTree.setLeft(head);
//            head = pRootOfTree;
//        }
//        ConvertSub(pRootOfTree.getRight());
//    }

    /**
     * 方法三：改进递归版
     * 解题思路：
     * 思路与方法二中的递归版一致，仅对第2点中的定位作了修改，新增一个全局变量记录左子树的最后一个节点。
     */
    // 记录子树链表的最后一个节点，终结点只可能为只含左子树的非叶节点与叶节点
    protected static BiTreeNode leftLast = null;
    public static BiTreeNode Convert_1(BiTreeNode root) {
        if(root==null)
            return null;
        if(root.getLeft()==null&&root.getRight()==null){
            leftLast = root;// 最后的一个节点可能为最右侧的叶节点
            return root;
        }
        // 1.将左子树构造成双链表，并返回链表头节点
        BiTreeNode left = Convert_1(root.getLeft());
        // 3.如果左子树链表不为空的话，将当前root追加到左子树链表
        if(left!=null){
            leftLast.setRight(root);
            root.setLeft(leftLast);
        }
        leftLast = root;// 当根节点只含左子树时，则该根节点为最后一个节点
        // 4.将右子树构造成双链表，并返回链表头节点
        BiTreeNode right = Convert_1(root.getRight());
        // 5.如果右子树链表不为空的话，将该链表追加到root节点之后
        if(right!=null){
            right.setLeft(root);
            root.setRight(right);
        }
        return left!=null?left:root;
    }


    /**
     * 测试
     */
    static class Test{
        public static void printDoubleList(BiTreeNode listHead){
            BiTreeNode pNode = listHead;
            System.out.println();
            System.out.println("从左到右：");
            while(pNode != null){
                System.out.print(pNode.getValue()+" ");
                if(pNode.getRight() == null)
                    break;
                pNode = pNode.getRight();
            }

            System.out.println();
            System.out.println("从右到左：");
            while(pNode != null){
                System.out.print(pNode.getValue()+" ");
                if(pNode.getLeft() == null)
                    break;
                pNode = pNode.getLeft();
            }
            System.out.println();
        }

        public static void test(String testName, BiTreeNode root){
            if(testName != null)
                System.out.println(testName+"begins:");

            BiTreeUtils.inOrderTraverse(root);

//            BiTreeNode listHead = ConvertBSTToBiList(root);
            BiTreeNode listHead = Convert(root);
//            BiTreeNode listHead = Convert_1(root);

            printDoubleList(listHead);
        }

        //            10
        //         /      \
        //        6        14
        //       /\        /\
        //      4  8     12  16
        public static void test1(){
            BiTreeNode root = null;
            System.out.println("请输入树：");
            root = BiTreeUtils.createBiTreePreOrder(root);

            test("test1", root);
        }

        //               5
        //              /
        //             4
        //            /
        //           3
        //          /
        //         2
        //        /
        //       1
        public static void test2(){
            BiTreeNode root = null;
            System.out.println("请输入树：");
            root = BiTreeUtils.createBiTreePreOrder(root);

            test("test2", root);
        }

        // 1
        //  \
        //   2
        //    \
        //     3
        //      \
        //       4
        //        \
        //         5

        public static void test3(){
            BiTreeNode root = null;
            System.out.println("请输入树：");
            root = BiTreeUtils.createBiTreePreOrder(root);

            test("test3", root);
        }

        //树中只有一个节点
        public static void test4(){
            BiTreeNode root = null;
            System.out.println("请输入树：");
            root = BiTreeUtils.createBiTreePreOrder(root);

            test("test4", root);
        }

        //树中没有节点
        public static void test5(){
            test("test5", null);
        }
    }
}
