package Offer.test_46;

//求1...n的和
public class SumFrom1ToN {
    public static void main(String[] args) {
        System.out.println(sumSolution(5));
        System.out.println(sumSolution_1(0));
    }

    /**
     * 使用逻辑与的短路特性
     */
    public static int sumSolution(int num){
        int sum = num;
        boolean b = (num > 0) && ((sum += sumSolution(num - 1)) > 0);
        return sum;
    }

    /**
     * 使用异常机制退出递归
     */
    public static int sumSolution_1(int num){
        try {
            int i = 1 / num;
            return num + sumSolution_1(num - 1);
        }catch (Exception e){
            return 0;
        }
    }
}
