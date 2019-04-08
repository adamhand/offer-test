package Offer.test54;

public class IsNumic_2 {
    // 数组下标成员变量
    private static int index;

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


    public static boolean isNumeric(char[] str) {
        // 输入异常
        if (str == null)
            return false;
        index = 0;
        // 正负号开头
        if (str[index] == '+' || str[index] == '-')
            index++;
        if (index == str.length)
            return false;
        // 设置numeric判断是否为数字
        boolean numeric = true;
        scanDigits(str);
        if (index != str.length) {
            // 小数
            if (str[index] == '.') {
                index++;
                scanDigits(str);
                if (index < str.length && (str[index] == 'e' || str[index] == 'E'))
                    numeric = isExponential(str);
            } else if (str[index] == 'e' || str[index] == 'E')
                numeric = isExponential(str);
            else
                // 出现了异常字符
                numeric = false;
        }
        return numeric && index == str.length;
    }

    // 扫描数组，如果当前字符为数字，index++
    private static void scanDigits(char[] str) {
        while (index < str.length && str[index] >= '0' && str[index] <= '9')
            index++;
    }

    // 判断是否为科学计数法表示的数值的结尾部分
    private static boolean isExponential(char[] str) {
        if (str[index] != 'e' && str[index] != 'E')
            return false;
        index++;
        if (index == str.length)
            return false;
        if (str[index] == '+' || str[index] == '-')
            index++;
        if (index == str.length)
            return false;
        scanDigits(str);
        // 如果存在特殊字符，index不会为str.length
        return index == str.length ? true : false;
    }
}
