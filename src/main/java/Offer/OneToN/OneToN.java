package Offer.OneToN;

//打印1到最大的n位数
public class OneToN {
    public static void main(String[] args) {
        print1ToMaxOfN(2);
    }

    public static void print1ToMaxOfN(int n){
        if(n < 0)
            return;
        StringBuilder number = new StringBuilder();
        System.out.println(number.length());
        for(int i = 0; i < n; i++)
            number.append('0');

        while(!increment(number)){
            printNumber(number);
        }
    }

    public static boolean increment(StringBuilder sb){
        boolean isOverflow = false;        //是否最高位溢出
        int nTakeOver = 0;                 //进位
        for(int i = sb.length() - 1; i >= 0; i--){
            int nSum = sb.charAt(i) - '0' + nTakeOver; //某位上的数加上进位
            if(i == sb.length() - 1)  //如果当前位为最低位，最低位加一
                nSum++;
            if(nSum >= 10){          //要产生进位
                if(i == 0)
                    isOverflow = true;
                else {
                    nSum -= 10;
                    nTakeOver = 1; //进位
                    sb.setCharAt(i, (char)('0'+nSum));
                }
            }
            else {
                sb.setCharAt(i, (char)('0'+ nSum));  //不产生进位，也就不需要运算高位，直接跳出
                break;
            }
        }

        return isOverflow;
    }

    public static void printNumber(StringBuilder sb){
        boolean isBegin0 = true;
        //因为字符串前面补得是'0'，打印时要跳过这些0
        for(int i = 0; i < sb.length(); i++){
            if(isBegin0 && sb.charAt(i) != '0')
                isBegin0 = false;
            if(!isBegin0)
                System.out.print(sb.charAt(i));
        }
        System.out.print(" ");
    }
}
