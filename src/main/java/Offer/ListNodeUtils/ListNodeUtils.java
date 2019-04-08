package Offer.ListNodeUtils;

import java.util.Scanner;

public class ListNodeUtils {

    public static void main(String[] str){
//        ListNode head = createList(10);
        ListNode head = createList();
        printList(head);
    }

    /**
     * 注意是逆序建立链表
     * @return
     */
    public static ListNode createList(){
        Scanner scanner = new Scanner(System.in);
        ListNode head = new ListNode(-1);
        head.next = null;
        int length = 0;
        int value = 0;

        System.out.println("请输入链表的长度：");
        length = scanner.nextInt();

        System.out.println("请依次输入链表的值：");
        for(int i = 0; i < length; i++){
            value = scanner.nextInt();
            ListNode node = new ListNode();
            node.value = value;
            node.next = head.next;
            head.next = node;
        }
        return head;
    }

    public static ListNode createList(int length){
        ListNode head = new ListNode(-1);
        head.next = null;

        for(int i = 0; i < length; i++){
            ListNode node = new ListNode();
            node.value = i;
            node.next = head.next;
            head.next = node;
        }

        return head;
    }

    public static void printList(ListNode head){
        if(head == null)
            return;
        ListNode cur = head;
        while(cur.next != null){
            System.out.print(cur.next.value+" ");
            cur = cur.next;
        }
        System.out.println();
    }

}
