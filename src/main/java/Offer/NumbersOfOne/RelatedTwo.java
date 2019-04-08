package Offer.NumbersOfOne;

//相关题目2
public class RelatedTwo {
    public static void main(String[] args) {
        System.out.println(changeNumber(10, -7));
        System.out.println(Double.compare(0.0, 0.000000001));
    }

    public static int changeNumber(int m, int n){
        int count = 0; //需要改变的次数

        int xor = m ^ n;
        System.out.println(xor);
        while(xor != 0){
            count++;
            xor &= (xor - 1);
        }
        return count;
    }
}
