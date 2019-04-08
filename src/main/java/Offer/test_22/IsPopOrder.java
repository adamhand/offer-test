package Offer.test_22;

import java.util.Stack;

//判断一个序列是不是压栈顺序的弹出序列
public class IsPopOrder {
    public static boolean isAPopOrder(int[] pushSequence, int[] popSequence){
        if(pushSequence == null || popSequence == null)
            return false;
        Stack<Integer> stack = new Stack<>();
        int n = pushSequence.length;
        for(int pushInx = 0, popInx = 0; pushInx < n; pushInx++){
            stack.push(pushSequence[pushInx]);
            while(popInx < n && !stack.isEmpty() && stack.peek() == popSequence[popInx]){
                stack.pop();
                popInx++;
            }
        }
        return stack.isEmpty();
    }

    public static void main(String[] args){
        TestDemo.test1();
        TestDemo.test2();
        TestDemo.test3();
        TestDemo.test4();
        TestDemo.test5();
        TestDemo.test6();
        TestDemo.test7();
    }

    static class TestDemo{
        public static void test(String testName, int[] push, int[] pop, boolean expected){
            if(testName != null)
                System.out.println("begins:"+testName);
            if(isAPopOrder(push, pop) == expected)
                System.out.println("Passed");
            else
                System.out.println("Failed");
        }

        public static void test1(){
            int push[] = {1, 2, 3, 4, 5};
            int pop[] = {4, 5, 3, 2, 1};

            test("Test1", push, pop, true);
        }

        public static void test2(){
            int push[] = {1, 2, 3, 4, 5};
            int pop[] = {3, 5, 4, 2, 1};

            test("Test2", push, pop, true);
        }

        public static void test3(){
            int push[] = {1, 2, 3, 4, 5};
            int pop[] = {4, 3, 5, 1, 2};

            test("Test3", push, pop, false);
        }

        public static void test4(){
            int push[] = {1, 2, 3, 4, 5};
            int pop[] = {3, 5, 4, 1, 2};

            test("Test4", push, pop, false);
        }

        public static void test5(){
            int push[] = {1};
            int pop[] = {2};

            test("Test5", push, pop, false);
        }

        public static void test6(){
            int push[] = {1};
            int pop[] = {1};

            test("Test6", push, pop, true);
        }

        public static void test7(){
            test("Test7", null, null, false);
        }
    }
}
