package Offer.test_56;

class ListNode{
    public int value;
    public ListNode next;

    public ListNode(){}

    public ListNode(int value) {
        this.value = value;
    }
}

//单向链表中环的入口
public class EntryInALoop {
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

        System.out.println(findEntryNode_1(head).value);
    }

    /**
     * 先求出链表的长度k，则倒数第k个节点就是入口
     */
    //找到相遇点
    public static ListNode meetingNode(ListNode head){
        if(head == null)
            return null;
        ListNode slow = head.next;
        if(slow == null)
            return null;
        ListNode fast = slow.next;
        while(fast != null && slow != null){
            if(fast == slow)
                return fast;
            slow = slow.next;
            fast = fast.next;
            if(fast != null)
                fast = fast.next;
        }
        return null;
    }

    public static ListNode findEntryNode(ListNode head){
        ListNode meetingNode = meetingNode(head);
        if(meetingNode == null)
            return null;

        int nodeInLoop = 1;
        ListNode node1 = meetingNode;
        while(node1.next != meetingNode){
            node1 = node1.next;
            nodeInLoop++;
        }

        node1 = head;
        for(int i = 0; i < nodeInLoop; i++)
            node1 = node1.next;

        ListNode node2 = head;
        while(node1 != node2){
            node1 = node1.next;
            node2 = node2.next;
        }
        return node1;
    }

    /**
     * 方法二：不用求得环的长度
     */
    public static ListNode findEntryNode_1(ListNode head){
        ListNode meetingNode = meetingNode(head);
        if(meetingNode == null)
            return null;

        ListNode node1 = meetingNode;
        ListNode node2 = head;
        while(node1 != node2){
            node1 = node1.next;
            node2 = node2.next;
        }
        return node1;
    }
}
