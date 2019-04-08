package Offer.test_34;

//丑数
public class UglyNumber {
    public static void main(String[] args) {
        System.out.println(getUglyNumber(1500));
    }

    /**
     * 方法一：直接求
     */
//    private static boolean isUgly(int num){
//        if(num <= 0)
//            return false;
//        while (num % 2 == 0)
//            num /= 2;
//        while (num % 3 == 0)
//            num /= 3;
//        while (num % 5 == 0)
//            num /= 5;
//
//        return (num == 1) ? true : false;
//    }
//
//    //输入1500，找到第1500个丑数
//    public static int getUglyNumber(int num){
//        if(num <= 0)
//            return 0;
//        int number = 0;
//        int uglyFound = 0; //第几个丑数
//
//        while(uglyFound < num){
//            ++number;
//            if(isUgly(number))
//                ++uglyFound;
//        }
//        return number;
//    }

    /**
     * 用空间换时间
     */
    public static int getUglyNumber(int index){
        if(index <= 0)
            return 0;

        int[] uglynumbers = new int[index];
        uglynumbers[0] = 1;
        int nextUglyNumber = 1;

        int pMultiply2 = 0;
        int pMultiply3 = 0;
        int pMultiply5 = 0;

        while(nextUglyNumber < index){
            int minUglyNumber = minOfThree(uglynumbers[pMultiply2] * 2, uglynumbers[pMultiply3] * 3, uglynumbers[pMultiply5] * 5);
            uglynumbers[nextUglyNumber] = minUglyNumber;

            while(uglynumbers[pMultiply2] * 2 <= uglynumbers[nextUglyNumber])
                pMultiply2++;
            while(uglynumbers[pMultiply3] * 3 <= uglynumbers[nextUglyNumber])
                pMultiply3++;
            while(uglynumbers[pMultiply5] * 5 <= uglynumbers[nextUglyNumber])
                pMultiply5++;

            nextUglyNumber++;
        }
        return uglynumbers[index - 1];
    }

    private static int minOfThree(int a, int b, int c){
        int min = (a < b) ? a : b;
        min = (min < c) ? min : c;

        return min;
    }
}
