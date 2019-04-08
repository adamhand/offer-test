package Offer.DeleteOneNode;

class ListNode{
    int value;
    ListNode nextNode;
}

//删除链表中的一个节点
public class DeleteOneNodeInList {
    public static void main(String[] args) {
        ListNode head = new ListNode();
        head.value = 0;
        for(int i = 0; i < 5; i++){
            ListNode node = new ListNode();
            node.value = i;
            node.nextNode = head.nextNode;
            head.nextNode = node;
        }
        ListNode cur = head.nextNode;
        for(int i = 0; i < 5; i++){
            System.out.println(cur.value+" ");
            cur = cur.nextNode;
        }

        cur = head.nextNode;
        ListNode deleted = deleteNode(head,  cur);
        for(int i = 0; i < 5 ;i++)
            System.out.println(deleted.value+" ");
    }
    public static ListNode deleteNode(ListNode head, ListNode tobeDeleted){
        if(head == null || head.nextNode == null || tobeDeleted == null)
            return null;
        if(tobeDeleted.nextNode != null){
            ListNode next = tobeDeleted.nextNode;
            tobeDeleted.value = next.value;
            tobeDeleted.nextNode = next.nextNode;
        }else {
            ListNode cur = head;
            while(cur.nextNode != tobeDeleted)
                cur = cur.nextNode;
            cur.nextNode = null;
        }
        return head;
    }
}
