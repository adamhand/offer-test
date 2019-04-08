package Offer.test54;

//表示数值的字符串
//https://www.cnblogs.com/ggmfengyangdi/p/5812351.html
public class IsNumeric {
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
        if(isNumeric(str.toCharArray()) == expected)
            System.out.println(testName+"+Passed.");
        else
            System.out.println(testName+"+Failed.");
    }


    /**
     * 根据题目要求，正确的输入格式应该符合以下的形式：
     * [sign]integer-digits[.[fragment-digits]][e|E[sign]exponential-digits]
     * []中的表示不一定需要
     * @param str
     * @return
     */
    public static boolean isNumeric(char[] str) {
        if (str == null || str.length <= 0) return false;
        int index = 0;
        //判断是否有符号
        if(str[index] == '+' || str[index] == '-'){
            index++;
        }
        //如果只有符号位，则不是正确的数字
        if (index == str.length) return false;
        //扫描数字
        index = scanDigits(str,index);
        if (index < str.length){
            //如果有小数
            if (str[index] == '.'){
                index++;
                index = scanDigits(str,index);
                //如果有指数表示的形式
                if (index < str.length){
                    if (str[index] == 'e' || str[index] == 'E'){
                        index++;
                        return isExponential(str,index);
                    }
                    return false;
                }
                return true;
            }else if (str[index] == 'e' || str[index] == 'E'){
                //如果没有小数，且有指数形式
                index++;
                return isExponential(str,index);
            }
            return false;
        }
        return true;
    }

    private static boolean isExponential(char[] str, int index) {
        if (index < str.length){
            //如果是符号，跳到下一个
            if (str[index] == '+' || str[index] == '-'){
                index++;
            }
            index = scanDigits(str,index);
            if (index == str.length) return true;
            return false;
        }
        return false;
    }

    private static int scanDigits(char[] str, int index) {
        while (index < str.length && str[index] >= '0' && str[index] <= '9') index++;
        return index;
    }
}
