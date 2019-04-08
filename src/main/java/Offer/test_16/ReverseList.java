package Offer.test_16;

class ListNode{
    public int value;
    public ListNode next;

    public ListNode(){}

    public ListNode(int value) {
        this.value = value;
    }
}

//反转单向链表
public class ReverseList {
    public static void main(String[] args) {

    }

    public static ListNode reverseList(ListNode head){
        ListNode reverseHead = null;
        ListNode node = head;
        ListNode preNode = null;

        while(node != null){
            ListNode next = node.next;
            if(next == null)
                reverseHead = node;
            node.next = preNode;
            preNode = node;
            node = next;
        }
        return reverseHead;
    }
}
