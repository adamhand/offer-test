package Offer.test_52;

//构建乘积数组
public class Multiply {
    public static void main(String[] args) {

    }

    /**
     *矩阵法
     */
//    public int[] multiply(int[] A) {
//        if(A == null || A.length < 2)
//            return null;
//
//        int length = A.length;
//        int[] B = new int[length];
//        B[0] = 1;
//        //计算左三角
//        for(int i = 1; i < length; i++){
//            B[i] = B[i - 1] * A[i - 1];
//        }
//        //计算右三角   temp用来记录有三角每一行的值
//        int temp = 1;
//        for(int i = length - 2; i >= 0; i--){
//            temp *= A[i + 1];
//            B[i] *= temp;
//        }
//        return B;
//    }
// https://blog.csdn.net/zjkc050818/article/details/72800856

    /**
     * 两次累乘法
     */
    public int[] multiply(int[] A) {
        int n = A.length;
        int[] B = new int[n];
        for (int i = 0, product = 1; i < n; product *= A[i], i++)       /* 从左往右累乘 */
            B[i] = product;
        for (int i = n - 1, product = 1; i >= 0; product *= A[i], i--)  /* 从右往左累乘 */
            B[i] *= product;
        return B;
    }
}
