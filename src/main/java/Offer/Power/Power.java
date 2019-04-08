package Offer.Power;

public class Power {
    public static void main(String[] args) {
        System.out.println(Power(10, -2));
//        System.out.println(Power(10, -2));
//        System.out.println(Power(10, -3));
//        System.out.println(Power(10, 1));
//        System.out.println(Power(10, 3));
    }

    public static double Power(double base, int exponent){
        boolean isNegtive = false;
        //底数为0且指数为负数，为了避免对0求导数出现的错误，直接返回0.0
        if(Math.abs(base - 0.0) < Double.MIN_VALUE && exponent < 0){
            return 0.0;
        }
        if(exponent < 0){
            exponent = -exponent;
            isNegtive = true;
        }
        double result = powerWithUnsignedExponent(base, exponent);

        return isNegtive ? 1.0 / result : result;
    }

    private static double powerWithUnsignedExponent(double base, int exponent) {
        //0的0次方也输出1
        if(exponent == 0)
            return 1;
        if(exponent == 1)
            return base;

        //result相当于base^n/2
        //用移位运算代替除以2，效率高
        double result = powerWithUnsignedExponent(base, exponent >> 1);
        result *= result;
        //如果exponent为奇数，还要乘以base
        //用位运算代替求余操作(%)来判断一个数是奇数还是偶数
        if((exponent & 0x01) == 1)
            result *= base;

        return result;
    }

    private static double powerWithUnsignedExponent1(double base, int exponent){
        double result = 1.0;
        for(int i = 1; i <= exponent; i++)
            result *= base;

        return result;
    }
}
