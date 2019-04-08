package Offer.test54;

public class IsNumic_1 {
    private static int inx;

    public static void main(String[] args) {
        test("Test1", "100", true);
        test("Test2", "123.45e+6", true);
        test("Test3", "+500", true);
        test("Test4", "5e2", true);
        test("Test5", "3.1416", true);
        test("Test6", "600.", true);
        test("Test7", "-.123", true);
        test("Test8", "-1E-16", true);
        test("Test9", "1.79769313486232E+308", true);

        System.out.println();

        test("Test10", "12e", false);
        test("Test11", "1a3.14", false);
        test("Test12", "1+23", false);
        test("Test13", "1.2.3", false);
        test("Test14", "+-5", false);
        test("Test15", "12e+5.4", false);
        test("Test16", ".", false);
        test("Test17", ".e1", false);
        test("Test18", "+.", false);
    }

    public static void test(String testName, String str, boolean expected){
        if(isNumeric_2(str.toCharArray()) == expected)
            System.out.println(testName+"+Passed.");
        else
            System.out.println(testName+"+Failed.");
    }

    /**
     * 方法一：直接从头跟到尾比较
     * @param str
     * @return
     */
    public static boolean isNumeric(char[] str) {
        if(str==null || str.length==0){
            return false;
        }
        inx = 0;
        boolean flag = scanInteger(str);
        //判断小数部分
        if(inx<str.length && str[inx]=='.'){
            inx = inx+1;
            flag = scanUInteger(str)||flag;     //解释a,见代码下方
        }
        //判断指数部分
        if(inx<str.length && (str[inx]=='e' || str[inx]=='E')){
            inx = inx+1;
            flag = flag && scanInteger(str);
        }
        return flag && inx>=str.length;
    }

    //判断是否是整数
    public static boolean scanInteger(char[] str){
        if(inx<str.length &&(str[inx]=='+' || str[inx]=='-')){
            inx = inx+1;
        }
        return scanUInteger(str);
    }

    //判断是否是无符号整数
    public static boolean scanUInteger(char[] str){
        int inx1=inx;
        while(inx<str.length && str[inx]>='0' && str[inx]<='9'){
            inx = inx + 1;
        }
        return inx>inx1;
    }


    /**
     * 方法二：使用库函数
     */
    public static boolean isNumeric_1(char[] str){
        try {
            Double number = Double.parseDouble(new String(str));
        }catch (NumberFormatException e){
            return false;
        }
        return true;
    }

    /**
     * 方法三：使用正则表达式
     */
    public static boolean isNumeric_2(char[] str){
        if (str == null || str.length == 0)
            return false;
        return new String(str).matches("[+-]?\\d*(\\.\\d+)?([eE][+-]?\\d+)?");
    }
}
