package Offer.NumbersOfOne;

public class NumberOfOne {
    public static void main(String[] args) {
        System.out.println(numberOf1_1(1));
        System.out.println(numberOf1_1(0x7FFFFFFF));
        System.out.println(numberOf1_1(0x80000000)); //如果使用算数右移  >> 会产生死循环
        System.out.println(numberOf1_1(0xFFFFFFFF)); //如果使用算数右移  >> 死循环

        System.out.println(numberOf1_2(1));
        System.out.println(numberOf1_2(0x7FFFFFFF));
        System.out.println(numberOf1_2(0x80000000));
        System.out.println(numberOf1_2(0xFFFFFFFF));

        System.out.println(numberOf1_3(1));
        System.out.println(numberOf1_3(-7));
        System.out.println(numberOf1_3(0x7FFFFFFF));
        System.out.println(numberOf1_3(0x80000000));
        System.out.println(numberOf1_3(0xFFFFFFFF));

        System.out.println(numberOf1_4(1));
        System.out.println(numberOf1_4(-7));
        System.out.println(numberOf1_4(0x7FFFFFFF));
        System.out.println(numberOf1_4(0x80000000));
        System.out.println(numberOf1_4(0xFFFFFFFF));

    }

    public static int numberOf1_1(int n){
        int count;
        if(n < 0)      //去掉符号位
            count = -1;
        else
            count = 0;
        while(n != 0){
            if((n & 0x01) != 0){
                count++;
            }
            n >>>= 1;  //使用逻辑右移，可以避免死循环
            //n >>= 1;  //使用算数右移，输入为负数时会产生死循环
        }
        return count;
    }

    public static int numberOf1_2(int n){
        int count = 0;
        int flag = 1;
        while(flag > 0){
            if((n & flag) != 0){
                count++;
            }
            flag <<= 1;
        }
        return count;
    }

    public static int numberOf1_3(int n){
        int count = 0;
        if(n < 0)               //如果n小于0的话，要剪掉符号位的1
            count -= 1;
        while(n != 0){
            ++count;
            n &= (n - 1);
        }
        return count;
    }

    public static int numberOf1_4(int n){
        if(n < 0)      //去掉符号位
            return Integer.bitCount(n) - 1;
        else
            return Integer.bitCount(n);
    }
}
