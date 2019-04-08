package Offer.test_25;

import DataStructureAndAlgorithm.BiTree2.BiTreeNode;
import DataStructureAndAlgorithm.BiTree2.BiTreeUtils;

import java.util.Stack;

//二叉树中和为某一值的路径
//测试用例
//            10
//         /      \
//        5        12
//       /\
//      4  7
// 有两条路径上的结点和为22

//            10
//         /      \
//        5        12
//       /\
//      4  7
// 没有路径上的结点和为15

//               5
//              /
//             4
//            /
//           3
//          /
//         2
//        /
//       1
// 有一条路径上面的结点和为15

// 1
//  \
//   2
//    \
//     3
//      \
//       4
//        \
//         5
// 没有路径上面的结点和为16

// 树中只有1个结点
//void Test5()
//        {
//        BinaryTreeNode* pNode1 = CreateBinaryTreeNode(1);
//
//        printf("One path should be found in Test5.\n");
//        Test("Test5", pNode1, 1);
//
//        DestroyTree(pNode1);
//        }
//
//// 树中没有结点
//        void Test6()
//        {
//        printf("No paths should be found in Test6.\n");
//        Test("Test6", nullptr, 0);
//        }
public class FindPath {
    public static void main(String[] args) {
        BiTreeNode root = null;
        System.out.println("请输入树：");
        root = BiTreeUtils.createBiTreePreOrder(root);

        int sum = 16;
        findPath(root, sum);
    }



    //使用栈
    static Stack<Integer> path = new Stack<>();
    public static void findPath(BiTreeNode root, int sum){
        backtracking(root, sum, path);
    }

    public static void backtracking(BiTreeNode root, int target, Stack<Integer> stack){
        if(root == null)
            return;
        stack.push(root.getValue());
        target -= root.getValue();
        //符合条件就将路径输出
        if(target == 0 && root.getLeft() == null && root.getRight() == null){
            System.out.println("A path is found:");
            for(Integer i : stack)
                System.out.print(i+" ");
            System.out.println();
        }
        //如果不是叶子节点，则遍历它的子节点
        if(root.getLeft() != null)
            backtracking(root.getLeft(), target, stack);
        if(root.getRight() != null)
            backtracking(root.getRight(), target, stack);
        //返回父节点之前在路径上删除当前节点
        stack.pop();
    }
}
