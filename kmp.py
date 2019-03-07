class kmp():
    """kmp字符串匹配算法"""
    def __init__(self):
        self.next = [-1,]
        self.j = 0
        self.k = -1
        self.i = 0
        self.f = 0
        self.count = 0

    def next_l(self,msc = "ababacdaccef"):
        """求next列表保存当字符串不匹配位出现时，模式串下一位比较位的位置"""
        while (len(msc) - self.f - 1):
            if self.k == -1 or msc[self.f] == msc[self.k]:
                self.f = self.f + 1
                self.k = self.k + 1
                if msc[self.f] == msc[self.k]:
                    self.next.append(self.next[self.k])
                else:
                    self.next.append(self.k)
            else:
                # self.k = self.next[self.k]
                if msc[self.k] == msc[self.f]:
                    self.next.append(self.next[self.k])
                    self.k = 0
                    if msc[self.k] == msc[self.f]:
                        self.next.append(-1)
                    self.f = self.f  + 1
                else:
                    self.k = self.next[self.k]


        print(self.next)

    def mnpp(self,mnc = "aaaaabaababacdaccefbssabaababacdaccebacababababacdacceacdabababcababacdaccefabaababacdaccebacdabababacdabcc",msc = "ababacdaccef"):
        """根据模式串next列表，将模式串与主串匹配，并计数输出"""
        while 1:
            if msc[self.i] == mnc[self.j]:
                self.i = self.i + 1
                self.j = self.j + 1
            else:
                self.i = self.next[self.i]
                if self.i == -1:
                    self.j += 1
                    self.i = 0
            if self.i == len(msc):
                self.count = self.count + 1
                self.i = 0
            if self.j == len(mnc):
                break
        print(self.count)

if __name__ == "__main__":
    kmp = kmp()
    msc = "abcabcabc"
    mnc ="dsfabcfabcabcabcasfabcdabcabcabcdd"
    kmp.next_l(msc)
    kmp.mnpp(mnc,msc)
