package Offer.test_43;

import java.util.Arrays;

//n个骰子的点数
public class Probability {
    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();
        printProbability(11);
        long endTime = System.currentTimeMillis();
        System.out.println("第一个程序执行的时间为："+ (endTime - startTime));

        startTime = System.currentTimeMillis();
        printProbability_1(11);
        endTime = System.currentTimeMillis();
        System.out.println("第二个程序执行的时间为："+ (endTime - startTime));
    }

    /**
     * 使用递归的方法
     */
    private static final int gMaxValue = 6;
    //number表示骰子的数量
    public static void printProbability(int number){
        if(number < 1)
            return;
        int maxSum = number * gMaxValue;  //number个骰子点数的最大值
        int[] pProbabilities = new int[maxSum - number + 1];
        Arrays.fill(pProbabilities, 0);

        double total = Math.pow(gMaxValue, number);
        probability(number, pProbabilities);  //这个函数计算n~6n每种情况出现的次数
        for(int i = number; i <= maxSum; i++){
            double ratio = pProbabilities[i - number] / total;
            System.out.println("sum: "+ i + ", ratio: "+ ratio);
        }
    }

    private static void probability(int number, int[] pProbabilities){
        for(int i = 1; i <= gMaxValue; i++)  //从第一个骰子开始
            probability(number, number, i, pProbabilities);
    }
    //总共original个骰子，当前第 current个骰子，当前的和，贯穿始终的数组
    private static void probability(int original, int current, int sum, int[] pProbabilities){
        if(current == 1)
            pProbabilities[sum - original]++;
        else{
            for(int i = 1; i <= gMaxValue; i++)
                probability(original, current - 1, sum + i, pProbabilities);
        }
    }

    /**
     * 基于循环的解法
     */
    //基于循环求骰子点数
    public static void printProbability_1(int number){
        if(number<1)
            return;
        int[][] pProbabilities = new int[2][gMaxValue*number +1];
        for(int i = 0;i < gMaxValue; i++){//初始化数组
            pProbabilities[0][i] = 0;
            pProbabilities[1][i] = 0;
        }
        int flag = 0;
        for(int i = 1;i <= gMaxValue; i++){//当第一次抛掷骰子时，有6种可能，每种可能出现一次
            pProbabilities[flag][i] = 1;
        }
        //从第二次开始掷骰子，假设第一个数组中的第n个数字表示骰子和为n出现的次数，
        //在下一循环中，我们加上一个新骰子，此时和为n的骰子出现次数应该等于上一次循环中骰子点数和为n-1,n-2,n-3,n-4,n-5，
        //n-6的次数总和，所以我们把另一个数组的第n个数字设为前一个数组对应的n-1,n-2,n-3,n-4,n-5，n-6之和
        for(int k = 2; k <= number; k++){
            for(int i = 0; i < k; i++){//第k次掷骰子，和最小为k，小于k的情况是不可能发生的！所以另不可能发生的次数设置为0！
                pProbabilities[1 - flag][i] = 0;
            }
            for(int i = k;i <= gMaxValue * k; i++){//第k次掷骰子，和最小为k，最大为g_maxValue*k
                pProbabilities[1 - flag][i] = 0;//初始化，因为这个数组要重复使用，上一次的值要清0
                for(int j = 1;j <= i && j <= gMaxValue; j++){
                    pProbabilities[1 - flag][i] += pProbabilities[flag][i - j];
                }
            }
            flag = 1 - flag;
        }
        double total = Math.pow(gMaxValue, number);
        for(int i=number; i <= gMaxValue * number; i++){
            double ratio = pProbabilities[flag][i] / total;
            System.out.println("sum: "+i+", ratio: "+ ratio);
        }
    }
}
