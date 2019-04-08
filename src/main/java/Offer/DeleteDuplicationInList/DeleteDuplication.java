package Offer.DeleteDuplicationInList;

class Node {
    public int value;
    public Node next;

    public Node() {}

    public Node(int value) {
        this.value = value;
    }
}

public class DeleteDuplication {
    public static void main(String[] args) {
        test();
    }

    public static void test(){
        int[] num = { 2, 3, 3, 5, 7, 8, 8, 8, 9, 9, 10 };
        Node head = new Node();
        Node pre = head;
        for (int i = 0; i < num.length; i++) {
            Node node = new Node(num[i]);
            pre.next = node;
            pre = node;
        }
        System.out.print("删除重复结点前的链表：");
        print(head.next);
        deleteRepeteNode(head);
//        Node delete = deleteDuplication(head);
        System.out.print("删除重复结点后的链表：");
        print(head.next);
    }

    /**
     * 打印输出链表的结点值
     * @param head 链表头结点的下一个结点
     */
    public static void print(Node head) {
        while (head != null) {
            System.out.print(head.value + " ");
            head = head.next;
        }
        System.out.println();
    }

    /**
     * 删除链表中重复的结点值，只保留第一个重复结点的值
     * @param head 链表的头结点
     */
    public static void deleteRepeteNode(Node head) {
        Node pre = head.next;
        Node cur;
        while (pre != null) {
            cur = pre.next;
            if (cur != null && (pre.value == cur.value)) {
                pre.next = cur.next;
            } else {
                pre = cur;
            }
        }
    }

    public static void deleteDuplication(Node head){
        Node pre = head.next;
        Node cur = pre.next;
        while(pre != null && cur != null){
            if(pre.value == cur.value){
                while(cur.value == cur.next.value)
                    cur = cur.next;
                pre.next = cur.next;
                cur = pre.next;
            }else {
                pre = pre.next;
                cur = cur.next;
            }
        }
    }

//    public static Node deleteDuplication(Node pHead)
//    {
//        if(pHead == null||pHead.next == null)
//            return pHead;
//        if(pHead.value == pHead.next.value){//第一个节点是重复节点，则跳过重复节点
//            Node node = pHead.next;
//            while(node != null&&node.value == pHead.value)
//                node = node.next;
//            return deleteDuplication(node);
//        }else{
//            //第一个节点不是重复节点
//            pHead.next = deleteDuplication(pHead.next);
//            return pHead;
//        }
//    }

}
