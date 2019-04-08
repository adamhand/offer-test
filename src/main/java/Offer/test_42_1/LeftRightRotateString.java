package Offer.test_42_1;

public class LeftRightRotateString {
    public static void main(String[] args) {
        String str = "abcdefg";
        System.out.println(leftRotateString(null, 8));

        System.out.println(rightRotateString(null, 8));
    }

    //左旋转字符串，index表示左旋转的数量
    public static String leftRotateString(String str, int index){
        if(str == null || str.length() == 0)
            return null;
        if(index <= 0 || index > str.length())
            return str;
        char[] chars = str.toCharArray();

        //翻转字符串前index个字符
        reverse(chars, 0, index - 1);
        //翻转字符串后面的部分
        reverse(chars, index, str.length() - 1);
        //翻转整个字符串
        reverse(chars, 0, str.length() - 1);

        return String.valueOf(chars);
    }


    //右旋转字符串，index右旋转的数量
    public static String rightRotateString(String str, int index){
        if(str == null || str.length() == 0)
            return null;
        if(index <= 0 || index > str.length())
            return str;
        char[] chars = str.toCharArray();

        //翻转字符串后index个字符
        reverse(chars, str.length() - index, str.length() - 1);
        //翻转前面的字符
        reverse(chars, 0, str.length() - index - 1);
        //翻转整个字符串
        reverse(chars, 0, str.length() - 1);

        return String.valueOf(chars);
    }


    //字符串反转
    public static void reverse(char[] chars, int i, int j){
        if(chars == null || chars.length == 0)
            return;
        while(i < j){
            char ch = chars[i];
            chars[i] = chars[j];
            chars[j] = ch;

            i++;
            j--;
        }
    }
}
