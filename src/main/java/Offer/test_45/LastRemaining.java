package Offer.test_45;

import java.util.ArrayList;
import java.util.List;

//圆圈中最后剩下的数字
public class LastRemaining {
    public static void main(String[] args) {
        lastRemaining(5, 3);
        lastRemaining_Solution(5, 3);
    }

    /**
     * 采用链表模拟法
     */
    public static void lastRemaining(int n, int m){
        if(n < 1 || m < 1)
            return;
        List<Integer> list = new ArrayList<>();
        for(int i = 0; i < n; i++)
            list.add(i);
        int k = 0;
        while(list.size() > 1){
            k = k + m;
            k = k % (list.size()) - 1;
            if(k < 0){
                System.out.print(list.get(list.size() - 1)+" ");
                list.remove(list.size() - 1);
                k = 0;
            }else {
                System.out.print(list.get(k)+" " +
                        "");
                list.remove(list.get(k));
            }
        }
    }

    /**
     * 分析法
     */
    public static int lastRemaining_Solution(int n, int m) {
        if(n <= 0)
            return -1;
        int res = 0;
        for(int i = 2; i <= n; i++){
            res = (res + m) % i;
//            System.out.print(res + " ");
        }
        return res;
    }
    //https://blog.csdn.net/abc7845129630/article/details/52823135
}
