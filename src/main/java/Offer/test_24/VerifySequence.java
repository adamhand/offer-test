package Offer.test_24;

//判断某个序列是不是二叉搜索树的后序遍历序列
public class VerifySequence {
    public static boolean VerifySquenceOfBST(int[] sequence) {
        if (sequence == null || sequence.length == 0)
            return false;
        return verify(sequence, 0, sequence.length - 1);
    }

    private static boolean verify(int[] sequence, int first, int last) {
        //递归终止条件
        if (last - first <= 1)
            return true;
        int rootVal = sequence[last];
        int cutIndex = first;
        while (cutIndex < last && sequence[cutIndex] <= rootVal)
            cutIndex++;
        for (int i = cutIndex; i < last; i++)
            if (sequence[i] < rootVal)
                return false;
        return verify(sequence, first, cutIndex - 1) && verify(sequence, cutIndex, last - 1);
    }

    public static void main(String[] args) {
        Test.test1();
        Test.test2();
        Test.test3();
        Test.test4();
        Test.test5();
        Test.test6();
        Test.test7();
        Test.test8();
    }

    //测试代码
    static class Test{
        public static void test(String testName, int[] sequence,boolean expected){
            if(testName != null)
                System.out.println(testName+" begins");
            if(VerifySquenceOfBST(sequence) == expected)
                System.out.println("Passed");
            else
                System.out.println("Failed");
        }

        //            10
        //         /      \
        //        6        14
        //       /\        /\
        //      4  8     12  16
        public static void test1(){
            int data[] = {4, 8, 6, 12, 16, 14, 10};
            test("test1", data, true);
        }
        //           5
        //          / \
        //         4   7
        //            /
        //           6
        public static void test2(){
            int data[] = {4, 6, 7, 5};
            test("test2", data, true);
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
        public static void test3(){
            int data[] = {1, 2, 3, 4, 5};
            test("test3", data, true);
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
        public static void test4(){
            int data[] = {5, 4, 3, 2, 1};
            test("test4", data, true);
        }

        //只有一个节点
        public static void test5(){
            int[] data = {5};
            test("test5", data, true);
        }

        public static void test6(){
            int data[] = {7, 4, 6, 5};
            test("test6", data, false);
        }

        public static void test7(){
            int data[] = {4, 6, 12, 8, 16, 14, 10};
            test("test7", data, false);
        }

        public static void test8(){
            test("test8", null, false);
        }
    }
}
