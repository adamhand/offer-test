package Offer.test_41_1;

// 和为 S 的连续正数序列
public class FindContinuousSequence {
    public static void main(String[] args) {
        findContinuousSequence(9);
        System.out.println();
        findContinuousSequence(15);
        System.out.println();
        findContinuousSequence(0);
        System.out.println();
        findContinuousSequence(4);
        System.out.println();
        findContinuousSequence(3);
        System.out.println();
        findContinuousSequence(100);
    }

    public static void findContinuousSequence(int sum){
        if(sum < 3)
            return;
        int left = 1;
        int right = 2;
        int mid = (1 + sum) >> 1;
        int curSum = left + right;

        while(left < mid){
            if(sum == curSum)
                printSequence(left, right);

            while(curSum > sum && left < mid){
                curSum -= left;
                left++;
                if(curSum == sum)
                    printSequence(left, right);
            }
            right++;
            curSum += right;
        }
    }
    private static void printSequence(int left, int right){
        for(int i = left; i <= right; i++)
            System.out.print(i+" ");
        System.out.println();
    }
}
