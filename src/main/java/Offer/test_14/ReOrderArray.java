package Offer.test_14;

public class ReOrderArray {
    public static void main(String[] args) {
        int[] arr = {1, 2, 3, 4, 5, 90, 87, 65, 64, 100, 0, 9};
        for(int i = 0; i < arr.length; i++)
            System.out.print(arr[i]+" ");
        reOrderArray_1(arr);
        System.out.println();
        for(int i = 0; i < arr.length; i++)
            System.out.print(arr[i]+" ");
    }

    public static void reOrderArray(int[] arr){
        if(arr == null || arr.length < 0)
            return;
        int head = 0, tail = arr.length - 1;
        while(head < tail){
            while(head < tail && (arr[head] & 0x01) != 0)
                head++;
            while(head < tail && (arr[tail] & 0x01) == 0)
                tail--;
            if(head < tail){
                int temp = arr[head];
                arr[head] = arr[tail];
                arr[tail] = temp;
            }
        }
    }

    public static void reOrderArray_1(int[] nums) {
        // 奇数个数
        int oddCnt = 0;
        for (int val : nums)
            if ((val & 0x01) != 0)
                oddCnt++;
        int[] copy = nums.clone();
        int i = 0, j = oddCnt;
        for (int num : copy) {
            if ((num & 0x01) != 0)
                nums[i++] = num;
            else
                nums[j++] = num;
        }
    }
}
