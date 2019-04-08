package Offer.test_32;


//1-n中整数1出现的次数
public class NumOfOne {
    public static void main(String[] args) {
        int num = 12;
//        System.out.println(numOfOneFrom1ToN(num));
        for(int i = 0; i <= 534; i++)
            System.out.println(i);
    }

    /**
     * 不考虑时间效率的解法
     * @param n
     * @return
     */
//    public static int numOfOneFrom1ToN(int n){
//        int number = 0;
//        //检查不大于n的所有整数
//        for(int i = 1; i <= n; i++)
//            number += numOf1(i);
//        return number;
//    }
//
//    private static int numOf1(int n){
//        int number = 0;
//
//        //检查数的每一位是不是为1
//        while (n != 0) {
//            if ((n % 10) == 1)
//                number++;
//            n /= 10;
//        }
//        return number;
//    }

    /**
     * 分析数字规律的解法
     */
    public static int numOfOneFrom1ToN(int n){
        if(n < 1)
            return 0;

        int count = 0;
        int base = 1;
        int round = n;

        while(round > 0){
            int weight = round % 10;
            round /= 10;
            count += round * base;
            if(weight == 1)
                count+=(n % base) + 1;
            else if(weight > 1)
                count += base;
            base *= 10;
        }
        return count;
    }
}
