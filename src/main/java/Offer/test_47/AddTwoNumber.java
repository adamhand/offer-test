package Offer.test_47;

//不使用运算符计算两个数的和
public class AddTwoNumber {
    public static void main(String[] args) {
        System.out.println(addTowNumber(9, 0));
        System.out.println(addTowNumber(-1, 0));
        System.out.println(addTowNumber(-1, 4));


        System.out.println(addTowNumbers_1(0, 9));
        System.out.println(addTowNumbers_1(-1, 9));
        System.out.println(addTowNumbers_1(-1, 0));
    }

    /**
     * 使用循环
     */
    public static int addTowNumber(int a, int b){
        int sum = 0;
        int carry = 0;
        do{
            sum = a ^ b;
            carry = (a & b) << 1;

            a = sum;
            b = carry;
        }while (b != 0);

        return a;
    }

    /**
     * 使用递归
     */
    public static int addTowNumbers_1(int a, int b) {
        return b == 0 ? a : addTowNumbers_1(a ^ b, (a & b) << 1);
    }
}
