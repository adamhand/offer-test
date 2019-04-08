package Offer.NumbersOfOne;

//判断一个数是否为2的整数次方
public class IsIntegralMultipleOf2 {
    public static void main(String[] args) {
        System.out.println(isIntegerMultipleOf2(2));
        System.out.println(isIntegerMultipleOf2(3));
        System.out.println(isIntegerMultipleOf2(4));
        System.out.println(isIntegerMultipleOf2(8));
    }

    public static boolean isIntegerMultipleOf2(int num){
        if((num & (num - 1)) == 0)
            return true;
        else
            return false;
    }
}
