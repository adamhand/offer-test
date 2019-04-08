package Offer.test_null2;

import java.util.LinkedList;
import java.util.Queue;

//礼物的最大值
public class MaxValueOfGifts {
    public static void main(String[] args) {
        int[][] data = {
                {1,10,3,8},
                {12,2,9,6},
                {5,7,4,11},
                {3,7,16,5}};
//        System.out.println(getMaxVaule(data));
        System.out.println(getMaxVaule2(null));
    }

    /**
     * 动态规划
     */
    //方法一：动态规划
//    public static int getMaxVaule(int[][] data){
//        if(data == null || data.length == 0 || data[0].length == 0)
//            return 0;
//        int[][] dp = new int[2][data[0].length];
//        int dpCurRowIndex = 0, dpPreRowIndex = 0;
//        for(int row = 0; row < data.length; row++){
//            dpCurRowIndex = row & 1;
//            dpPreRowIndex = 1 - dpCurRowIndex;
//            for(int col = 0; col < data[0].length; col++){
//                if(col == 0)
//                    dp[dpCurRowIndex][col] = dp[dpPreRowIndex][col] + data[row][col];
//                else{
//                    if(dp[dpPreRowIndex][col] >= dp[dpCurRowIndex][col - 1])
//                        dp[dpCurRowIndex][col] = dp[dpPreRowIndex][col] + data[row][col];
//                    else
//                        dp[dpCurRowIndex][col] = dp[dpCurRowIndex][col - 1]+data[row][col];
//                }
//            }
//        }
//        return dp[(data.length - 1) & 1][data[0].length - 1];
//    }

    //二位数组进行辅助
//    public int getMaxValue(int[][] arr) {
//        if (arr == null || arr.length == 0) {
//            return 0;
//        }
//
//        int rows = arr.length;
//        int cols = arr[0].length;
//
//        int[][] maxValue = new int[rows][cols];
//
//        for (int i = 0; i < rows; i++) {
//            for (int j = 0; j < cols; j++) {
//                int left = 0;
//                int up = 0;
//                if(i>0){
//                    up = maxValue[i-1][j];
//                }
//                if(j>0){
//                    left = maxValue[i][j-1];
//                }
//                maxValue[i][j] = Math.max(up, left) + arr[i][j];
//            }
//        }
//        return maxValue[rows-1][cols-1];
//    }

    //一维数组进行辅助
    public int getMaxValue(int[][] arr) {
        if (arr == null || arr.length == 0) {
            return 0;
        }

        int rows = arr.length;
        int cols = arr[0].length;

        int[][] maxValue = new int[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int left = 0;
                int up = 0;
                if(i>0){
                    up = maxValue[i-1][j];
                }
                if(j>0){
                    left = maxValue[i][j-1];
                }
                maxValue[i][j] = Math.max(up, left) + arr[i][j];
            }
        }
        return maxValue[rows-1][cols-1];
    }

    /**
     * 用图的广度优先搜索
     */
    //有向图的遍历（广度优先，可再剪枝进行优化）
    public static int getMaxVaule2(int[][] data){
        if(data == null || data.length == 0 || data[0].length == 0)
            return 0;
        int maxRowIndex = data.length - 1;
        int maxColIndex = data[0].length - 1;
        Queue<Node> queue = new LinkedList<>();
        queue.offer(new Node(0,0,data[0][0]));
        Node nodeCur = null;
        while (!(queue.peek().row == maxRowIndex && queue.peek().col == maxColIndex)){
            nodeCur = queue.poll();
            if(nodeCur.row != maxRowIndex)
                queue.offer(new Node(nodeCur.row + 1,nodeCur.col,nodeCur.sum + data[nodeCur.row + 1][nodeCur.col]));
            if(nodeCur.col!=maxColIndex)
                queue.offer(new Node(nodeCur.row,nodeCur.col + 1,nodeCur.sum+data[nodeCur.row][nodeCur.col + 1]));
        }
        int maxSum = 0,temp = 0;
        while (!queue.isEmpty()){
            temp = queue.poll().sum;
            if(temp > maxSum)
                maxSum = temp;
        }
        return maxSum;
    }
    public static class Node{
        int row;
        int col;
        int sum;
        public Node(int r,int c,int s){
            row = r;col = c;sum = s;
        }
    }
}
