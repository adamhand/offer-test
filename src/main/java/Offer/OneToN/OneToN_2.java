package Offer.OneToN;

public class OneToN_2 {
    public static void main(String[] args) {
        Print1ToMaxOfNDigits_3(5);
    }
    public static void Print1ToMaxOfNDigits_3(int n){
        if(n < 0){
            return;
        }
        StringBuffer s = new StringBuffer(n);
        for(int i = 0; i < n; i++){
            s.append('0');
        }
        for(int i = 0; i < 10; i++){
            s.setCharAt(0, (char) (i+'0'));
            Print1ToMaxOfNDigits_3_Recursely(s, n, 0);
        }

    }
    public static void Print1ToMaxOfNDigits_3_Recursely(StringBuffer s, int n , int index){
        if(index == n - 1){
            PrintNumber(s);
            return;
        }

        for(int i = 0; i < 10; i++){
            //设置某一位的数
            s.setCharAt(index+1, (char) (i+'0'));
            //第一次递归设置十位数，第二次设置百位数......
            Print1ToMaxOfNDigits_3_Recursely(s, n, index+1);
        }
    }
    public static void PrintNumber(StringBuffer s){
        boolean isBeginning0 = true;
        for(int i = 0; i < s.length(); i++){
            if(isBeginning0 && s.charAt(i) != '0'){
                isBeginning0 = false;
            }
            if(!isBeginning0){
                System.out.print(s.charAt(i));
            }
        }

        System.out.println();
    }
}

