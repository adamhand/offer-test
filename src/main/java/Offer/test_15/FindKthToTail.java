package Offer.test_15;

import java.util.HashSet;

class ListNode{
    public int value;
    public ListNode next;

    public ListNode(){}

    public ListNode(int value) {
        this.value = value;
    }
}

//找到链表倒数第k个节点
public class FindKthToTail {
    public static void main(String[] args) {
        ListNode head = new ListNode();
        head.next = null;
        head.value = 0;
        for(int i = 0; i < 10; i++){
            ListNode node = new ListNode();
            node.value = i;
            node.next = head.next;
            head.next = node;
        }
        ListNode cur = head.next;
        for(int i = 0; i < 10; i++){
            System.out.print(cur.value+" ");
            cur = cur.next;
        }
        System.out.println();
//        System.out.println(Solution(head, 3).value);
//        System.out.println(Solution(head, 10).value);
//        System.out.println(Solution(head, 0).value);
//        System.out.println(Solution(head, 5).value);


//        System.out.println(findCenterNode(head).value);

        System.out.println(hasCycleOrNot(head));
    }

    public static ListNode Solution(ListNode head, int k){
        if(head == null || k == 0)
            return null;
        ListNode ahead = head;
        ListNode behind = null;

        for(int i = 0; i < k - 1; i++){
            if(ahead.next != null)
                ahead = ahead.next;
            else
                return null;
        }
        behind = head;
        while(ahead.next != null){
            ahead = ahead.next;
            behind = behind.next;
        }
        return behind;
    }

    //找到链表的中间节点
    public static ListNode findCenterNode(ListNode head){
        //链表为空或只有头结点
        if(head == null || head.next == null)
            return null;
        //链表只存在一个节点
        if(head.next.next == null)
            return head.next;
        ListNode fast = head;
        ListNode slow = head;
        while(fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    /**
     * 判断单向链表中是否有环
     */
    //方法1：申请空间
    public static boolean hsCycleOrNot(ListNode head){
        if(head == null)
            return false;
        HashSet<Integer> set = new HashSet<>();
        ListNode cur = head;
        while(cur.next != null){
            if(set.contains(cur.next.value))
                return false;
            else
                set.add(cur.next.value);
            cur = cur.next;
        }
        return true;
    }

    //方法2：使用两个指针。
    public static boolean hasCycleOrNot(ListNode head){
        if(head == null)
            return true;
        ListNode fast = head;
        ListNode slow = head;
        while(fast.next != null){
            if(fast.next.next != null)
                fast = fast.next.next;
            else
                return true;

            if(fast == slow)
                return false;
            slow = slow.next;
        }
        return true;
    }
}
