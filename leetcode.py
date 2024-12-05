import math
from functools import cache
from typing import List

# =============================================================================
# 2418. Sort the People

# names = ["Mary", "John", "Emma"]
# heights = [180, 165, 170]

def sortPeople(self, names, heights):
    dic = {}
    new_list = []
    for i in range(len(names)):
        dic[heights[i]] = names[i]
    dic = sorted(dic.items(), reverse=True)
    for i in dic:
        new_list.append(i[1])
    return new_list

def sortPeople(self, names: List[str], heights: List[int]) -> List[str]:
    persons = []
    for i in range(len(names)):
        persons.append({"name": names[i], "height": heights[i]})

    persons.sort(key=lambda x: x["height"], reverse=True)
    result = [person["name"] for person in persons]
    return result

# =============================================================================
# 2942. Find Words Containing Character

def findWordsContaining(self, words, x):
    res = []
    for i in range(len(words)):
        if x in words[i]:
            res.append(i)
    return res

# =============================================================================
# 1684. Count the Number of Consistent Strings

def countConsistentStrings(allowed, words):
    count = 0
    for word in words:
        match = True
        for letter in word:
            if letter not in allowed:
                match = False
        if match:
            count += 1
    print(count)

# =============================================================================
# 1662. Check If Two String Arrays are Equivalent

def arrayStringsAreEqual(self, word1: List[str], word2: List[str]) -> bool:
    return ''.join(word1) == ''.join(word2)

def arrayStringsAreEqual(self, word1, word2):
    new1 = ''
    new2 = ''
    for i in word1:
        new1 += i
    for i in word2:
        new2 += i
    if new1 == new2:
        return True

# =============================================================================
# 344. Reverse String

def reverseString(s):
    new = ''.join(s)
    new = new[::-1]
    res = []
    for i in new:
        res.append(i)
    print(res)

def reverseString1(s):
    s[:] = s[::-1]
    print(s)

def reverseString3(self, s):
    s.reverse()
    print(s)

# =============================================================================
# 2710. Remove Trailing Zeros From a String

def removeTrailingZeros(num):
    return str(int(num[::-1]))[::-1]

# =============================================================================
# 1929. Concatenation of Array

def getConcatenation(self, nums):
    return nums * 2

def getConcatenation(self, nums):
    return nums + nums

def getConcatenation(self, nums):
    nums.extend(nums)
    return nums

# =============================================================================
# 1672. Richest Customer Wealth

def maximumWealth(self, accounts):
    res = []
    for i in accounts:
        res.append(sum(i))
    return max(res)

# =============================================================================
# 2798. Number of Employees Who Met the Target
hours = [0, 1, 2, 3, 4]
target = 2

# Output: 3

def numberOfEmployeesWhoMetTarget(hours, target):
    count = 0
    for i in hours:
        if i >= target:
            count += 1
    return count

# =============================================================================
# 1678. Goal Parser Interpretation
# G -> G
# () -> o
# (al) -> al
command = "G()(al)"

# Output: "Goal"
def interpret(self, command):
    command = command.replace('()', 'o')
    command = command.replace('(al)', 'al')
    return command

def interpret(self, command):
    return command.replace('()', 'o').replace('(al)', 'al')

# =============================================================================
# 1480. Running Sum of 1d Array

# 0)
def runningSum(self, nums):
    res = []
    curr = 0
    for elem in nums:
        curr += elem
        res.append(curr)
    return res

# # 1)
# def sum(self, nums, index):
#     sum = 0
#     for i in range(index + 1):
#         sum = sum + nums[i]
#     return sum

def runningSum(self, nums):
    res = []
    for x in range(len(nums)):  # 10
        elem = nums[x]
        res.append(self.sum(nums, x))
    return res

# 2)

def runningSum(self, nums):
    res = []
    curr = 0
    for x in range(len(nums)):
        elem = nums[x]
        curr += elem
        res.append(curr)
    return res

# 3)
def runningSum(nums):
    res = [nums[0]]
    for x in range(1, len(nums)):
        elem = nums[x]
        new = res[x - 1]
        new += elem
        res.append(new)
    return res

# 4)
def runningSum(self, nums):
    res = [nums[0]]
    for x in range(1, len(nums)):
        elem = nums[x]
        new = res[-1] + elem
        res.append(new)
    return res

# 4)
def runningSum(self, nums):
    res = []
    for x in range(len(nums)):
        elem = nums[x]
        if len(res) > 0:
            prev_sum = res[-1]
        else:
            prev_sum = 0
        res.append(prev_sum + elem)
    return res

# 5) перезапись, так не надо
def runningSum(nums):
    for i in range(1, len(nums)):
        nums[i] = nums[i - 1] + nums[i]
        return nums

# =============================================================================
# 1470. Shuffle the Array
def shuffle(self, nums, n):
    res = []
    for i in range(n):
        el = nums[i]
        sec = nums[i + n]
        res.append(el)
        res.append(sec)
    return res

# 2)

def shuffle(self, nums: List[int], n: int) -> List[int]:
    array1 = nums[:n]
    array2 = nums[n:]
    result = []
    for i in range(n):
        result.append(array1[i])
        result.append(array2[i])
    return result

# =============================================================================
# 1431. Kids With the Greatest Number of Candies
candies = [2, 3, 5, 1, 3]
extraCandies = 3

# Output: [true,true,true,false,true]

def kidsWithCandies(candies, extraCandies):
    res = []
    for i in range(len(candies)):
        el = candies[i] + extraCandies
        if max(candies) <= el:
            greatest = True
        else:
            greatest = False
        res.append(greatest)
    print(res)

def kidsWithCandies(self, candies, extraCandies):
    res = []
    max_el = max(candies)
    for candy in candies:
        el = candy + extraCandies
        if max_el <= el:
            res.append(True)
        else:
            res.append(False)
    return res

# =============================================================================
# 1920. Build Array from Permutation
'''Input: nums = [0,2,1,5,3,4]
Output: [0,1,2,4,5,3]
Explanation: The array ans is built as follows: 
ans = [nums[nums[0]], nums[nums[1]], nums[nums[2]], nums[nums[3]], nums[nums[4]], nums[nums[5]]]
    = [nums[0], nums[2], nums[1], nums[5], nums[3], nums[4]]
    = [0,1,2,4,5,3]'''

def buildArray(self, nums):
    res = []
    for i in range(len(nums)):
        el = nums[i]
        x = nums[el]
        res.append(x)
    return res

# =============================================================================
# 1281. Subtract the Product and Sum of Digits of an Integer

n = 564

def subtractProductAndSum(self, n):
    new = str(n)
    summ = 0
    product = 1
    for i in range(len(new)):
        el = int(new[i])
        summ += el
        product *= el
    return product - summ

def subtractProductAndSum(self, n):
    new = str(n)
    summ = 0
    product = 1
    for el in new:
        el = int(el)
        summ += el
        product *= el
    return product - summ

# =============================================================================
# 1512. Number of Good Pairs

def numIdenticalPairs(self, nums):
    res = 0
    for i in range(len(nums)):
        for j in range(len(nums)):
            if nums[i] == nums[j] and i < j:
                res += 1
    return res

# =============================================================================
# 1365. How Many Numbers Are Smaller Than the Current Number

def smallerNumbersThanCurrent(self, nums):
    res = []
    for i in range(len(nums)):
        count = 0
        for j in range(len(nums)):
            if j != i and nums[j] < nums[i]:
                count += 1
        res.append(count)
    return res

def smallerNumbersThanCurrent(self, nums):
    new = sorted(nums)
    res = []
    for i in nums:
        res.append(new.index(i))
    return res

# =============================================================================
# 1603. Design Parking System

# Input
# ["ParkingSystem", "addCar", "addCar", "addCar", "addCar"]
# [[1, 1, 0], [1], [2], [3], [1]]
# Output
# [null, true, true, false, false]

class ParkingSystem(object):
    def __init__(self, big, medium, small):
        self.big = big
        self.medium = medium
        self.small = small

    def addCar(self, carType):
        if carType == 1:
            if self.big > 0:
                self.big -= 1
                return True
            else:
                return False
        elif carType == 2:
            if self.medium > 0:
                self.medium -= 1
                return True
            else:
                return False
        else:
            if self.small > 0:
                self.small -= 1
                return True
            else:
                return False

# 2)

class ParkingSystem(object):
    def __init__(self, big, medium, small):
        self.available_slots = {
            1: big,
            2: medium,
            3: small
        }

    def addCar(self, carType):
        if self.available_slots[carType] > 0:
            self.available_slots[carType] -= 1
            return True
        return False

# Your ParkingSystem object will be instantiated and called as such:
# obj = ParkingSystem(big, medium, small)
# param_1 = obj.addCar(carType)

# =============================================================================
# 2769. Find the Maximum Achievable Number

def theMaximumAchievableX(self, num, t):
    return num + t * 2


# =============================================================================
# 1859. Sorting the Sentence

def sortSentence(self, s):
    s = s.split()
    s.sort(key=xyi)
    result = []
    for i in s:
        i = i[:-1]
        result.append(i)
    result = ' '.join(result)
    return result

def xyi(word):
    return word[-1]

# =============================================================================
# 2824. Count Pairs Whose Sum is Less than Target

def countPairs(self, nums, target):
    count = 0
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] < target:
                count += 1
    return count

def countPairs(self, nums, target):
    count = 0
    n = len(nums)
    for i in range(n):
        for j in range(n):
            if 0 <= i < j < n and nums[i] + nums[j] < target:
                count += 1
    return count

# =============================================================================
# 2160. Minimum Sum of Four Digit Number After Splitting Digits
num = 2932

# Output: 52

def minimumSum(self, num):
    new = sorted(str(num))
    return int(new[0] + new[2]) + int(new[1] + new[3])

def minimumSum(self, num):
    new = sorted(str(num))
    new1 = []
    new2 = []
    new1.extend([new[0], new[2]])
    new2.extend([new[1], new[3]])
    new1 = int(''.join(new1))
    new2 = int(''.join(new2))
    res = new1 + new2
    return res

# =============================================================================
# 1313. Decompress Run-Length Encoded List

def decompressRLElist(self, nums):
    res = []
    for index in range(0, len(nums), 2):
        freq = nums[index]
        val = nums[index + 1]
        res.extend([val] * freq)
    return res

# =============================================================================
# 2859. Sum of Values at Indices With K Set Bits

def sumIndicesWithKSetBits(self, nums, k):
    res = 0
    for i in range(len(nums)):
        i_bin = bin(i)
        count_1 = 0
        for x in i_bin:
            if x == '1':
                count_1 += 1
        if count_1 == k:
            res += nums[i]
    return res

# 2)
def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
    ans = 0
    n = len(nums)
    for i in range(n):
        if bin(i)[2:].count("1") == k:  ## checking set bits in binary num
            ans += nums[i]
    return (ans)

# =============================================================================
# 1720. Decode XORed Array

def decode(self, encoded, first):
    res = [first]
    for i in range(len(encoded)):
        x = res[i] ^ encoded[i]
        res.append(x)
    return res

# =============================================================================
# 1389. Create Target Array in the Given Order

def createTargetArray(self, nums, index):
    res = []
    for i in range(len(nums)):
        res.insert(index[i], nums[i])
    return res

def createTargetArray(self, nums, index):
    arr = []
    for n, i in zip(nums, index):
        arr.insert(i, n)
    return arr

# =============================================================================
# 1486. XOR Operation in an Array
n = 4
start = 3

# Output: 8

def xorOperation(n, start):
    res = start
    for i in range(1, n):
        res = res ^ (start + 2 * i)
    return res

# =============================================================================
# 1342. Number of Steps to Reduce a Number to Zero

def numberOfSteps(self, num):
    steps = 0
    while num != 0:
        if num % 2 == 0:
            num /= 2
            steps += 1
        else:
            num -= 1
            steps += 1
    return steps

# =============================================================================
# 2652. Sum Multiples

def sumOfMultiples(self, n):
    res = 0
    for i in range(3, n + 1):
        if i % 3 == 0 or i % 5 == 0 or i % 7 == 0:
            res += i
    return res

# =============================================================================
# 2520. Count the Digits That Divide a Number

def countDigits(self, num):
    res = 0
    for i in str(num):
        if num % int(i) == 0:
            res += 1
    return res

# =============================================================================
# 2535. Difference Between Element Sum and Digit Sum of an Array

def differenceOfSum(self, nums):
    sum1 = sum(nums)
    sum2 = 0
    for i in nums:
        for j in str(i):
            sum2 += int(j)
    return abs(sum1 - sum2)

# =============================================================================
# 1656. Design an Ordered Stream

class OrderedStream(object):
    def __init__(self, n):
        self.n = n
        self.new = [None] * n
        self.position = 0

    def insert(self, idKey, value):
        self.new[idKey - 1] = value
        res = []
        while self.position < self.n and self.new[self.position] is not None:
            res.append(self.new[self.position])
            self.position += 1
        return res

class OrderedStream:
    def __init__(self, n: int):
        self.pairs = dict()
        self.position = 0

    def insert(self, idKey: int, value: str) -> list[str]:
        self.pairs[idKey] = value

        res = list()
        while self.position + 1 in self.pairs:
            self.position += 1

            res.append(self.pairs[self.position])

        return res

# =============================================================================
# 2325. Decode the Message

class Solution(object):
    def decodeMessage(self, key, message):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        uniq_key = ''
        res = ''
        for i in key:
            if i not in uniq_key and i != ' ':
                uniq_key += i
        for i in message:
            if i == ' ':
                res += ' '
            else:
                x = uniq_key.index(i)
                res += alphabet[x]
        return res

class Solution(object):
    def decodeMessage(self, key, message):
        alphabet = ' abcdefghijklmnopqrstuvwxyz'  ## пробел заменяем на пробел
        uniq_key = ' '
        res = ''
        for i in key:
            if i not in uniq_key:
                uniq_key += i
        for i in message:
            x = uniq_key.index(i)
            res += alphabet[x]
        return res

class Solution:
    def decodeMessage(self, key: str, message: str) -> str:
        mapping = {' ': ' '}
        i = 0
        res = ''
        letters = 'abcdefghijklmnopqrstuvwxyz'

        for char in key:
            if char not in mapping:
                mapping[char] = letters[i]
                i += 1

        for char in message:
            res += mapping[char]

        return res

# =============================================================================
# 1913. Maximum Product Difference Between Two Pairs

def maxProductDifference(self, nums):
    new = sorted(nums)
    num1 = new[0] * new[1]
    num2 = new[-1] * new[-2]
    return num2 - num1

# =============================================================================
# 557. Reverse Words in a String III

def reverseWords(self, s):
    res = []
    new = s.split(' ')
    for i in new:
        i = i[::-1]
        res.append(i)
    res = ' '.join(res)
    return res

def reverseWords(self, s: str) -> str:
    s = s.split(' ')
    new = ''
    for word in s:
        new += word[::-1] + ' '

    return new[:-1]

# =============================================================================
# 2828. Check if a String Is an Acronym of Words

def isAcronym(self, words, s):
    res = False
    new = ''
    for i in words:
        new += i[0]
    if s == new:
        res = True
    return res

def isAcronym(self, words, s):
    new = ''
    for i in words:
        new += i[0]
    return s == new

# =============================================================================
# 804. Unique Morse Code Words

def uniqueMorseRepresentations(self, words):
    alph = 'abcdefghijklmnopqrstuvwxyz'
    morse = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--", "-.", "---",
             ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."]
    dic = {}
    for i in range(len(alph)):
        dic[alph[i]] = morse[i]
    new = []
    for word in words:
        new1 = ''
        for let in word:
            new1 += dic[let]
        new.append(new1)
    return len(set(new))

def uniqueMorseRepresentations(self, words):
    alph = 'abcdefghijklmnopqrstuvwxyz'
    morse = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--", "-.", "---",
             ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."]
    morse_dict = dict(zip(alph, morse))
    new = []
    for word in words:
        new1 = ''
        for let in word:
            new1 += morse_dict[let]
        new.append(new1)
    return len(set(new))

# =============================================================================
# 1732. Find the Highest Altitude

def largestAltitude(self, gain):
    res = 0
    new = 0
    for i in gain:
        new += i
        if new > res:
            res = new
    return res

def largestAltitude(self, gain: List[int]) -> int:
    highest_point = 0
    prev_altitude = 0
    for i in gain:
        prev_altitude += i
        highest_point = max(highest_point, prev_altitude)
    return highest_point

# =============================================================================
# 1464. Maximum Product of Two Elements in an Array

def maxProduct(self, nums):
    new = sorted(nums, reverse=True)
    res = (new[0] - 1) * (new[1] - 1)
    return res

def maxProduct(self, nums):
    new = sorted(nums)
    return (new[-1] - 1) * (new[-2] - 1)

# =============================================================================
# 1323. Maximum 69 Number

def maximum69Number(self, num):
    new = str(num)
    res = ''
    change = False
    for i in range(len(new)):
        if new[i] == '6':
            if change:
                res += '6'
            else:
                res += '9'
                change = True
        else:
            res += '9'
    return int(res)

def maximum69Number(self, num: int) -> int:
    return int(str(num).replace('6', '9', 1))

def maximum69Number(self, num):
    temp = num
    s = (str(num))
    for i in range(len(s)):
        if s[i] == "6":
            val = (int(s[:i] + "9" + s[i + 1:]))
        else:
            val = (int(s[:i] + "6" + s[i + 1:]))
        temp = max(temp, val)
    return temp

def maximum69Number(self, num):
    new = list(str(num))
    if '6' in new:
        idx = new.index('6')
        new[idx] = '9'
    return int(''.join(new))

# =============================================================================
# 2427. Number of Common Factors

def commonFactors(self, a, b):
    res = 0
    new = min(a, b)
    for i in range(1, new + 1):
        if a % i == 0 and b % i == 0:
            res += 1
    return res

# =============================================================================
# 728. Self Dividing Numbers

def selfDividingNumbers(self, left, right):
    res = []
    for i in range(left, right + 1):
        count = 0
        for j in str(i):
            if int(j) != 0 and int(i) % int(j) == 0:
                count += 1
        if count == len(str(i)):
            res.append(i)
    return res

class Solution(object):
    def isDividingNumber(self, number):
        for digit in str(number):
            if int(digit) == 0 or number % int(digit) != 0:
                return False
        return True

    def selfDividingNumbers(self, left, right):
        res = []
        for number in range(left, right + 1):
            if self.isDividingNumber(number):
                res.append(number)
        return res

# =============================================================================
# 1716. Calculate Money in Leetcode Bank

def totalMoney(self, n):
    res = 0
    save = 0
    start = 0
    for i in range(n):
        if i % 7 == 0:
            start += 1
            save = start
        else:
            save += 1
        res += save
    return res

# =============================================================================
# 2367. Number of Arithmetic Triplets

def arithmeticTriplets(self, nums, diff):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            for k in range(j + 1, len(nums)):
                if nums[j] - nums[i] == diff and nums[k] - nums[j] == diff:
                    count += 1
    return count

# =============================================================================
# 2006. Count Number of Pairs With Absolute Difference K

def countKDifference(self, nums, k):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] - nums[j] == k or nums[i] - nums[j] == k * -1:
                count += 1
    return count

def countKDifference(self, nums, k):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if abs(nums[i] - nums[j]) == k:  # abs() - модуль
                count += 1
    return count

def countKDifference(self, nums, k):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] - nums[j] == k or nums[j] - nums[i] == k:
                count += 1
    return count

# =============================================================================
# 2574. Left and Right Sum Differences

def leftRightDifference(self, nums):
    res = []
    sum_left = 0
    sum_rigth = sum(nums)
    for i in nums:
        sum_rigth -= i
        res.append(abs(sum_left - sum_rigth))
        sum_left += i
    return res

# так писать нельзя
def leftRightDifference(self, nums):
    res = []
    for i in range(len(nums)):
        left = self.calcSum(nums, 0, i - 1)
        right = self.calcSum(nums, i + 1, len(nums))
        res.append(abs(right - left))
    return res

def calcSum(self, nums, left, right):
    return sum(nums[max(left, 0):min(len(nums), right) + 1])

# =============================================================================
# 2974. Minimum Number Game

def numberGame(self, nums):
    new = sorted(nums)
    res = []
    for i in range(0, len(nums), 2):
        res.append(new[i + 1])
        res.append(new[i])
    return res

# =============================================================================
# 1688. Count of Matches in Tournament

def numberOfMatches(self, n):
    total = 0
    teams = n
    while teams > 1:
        if teams % 2 != 0:
            total += (teams - 1) / 2
            teams = (teams - 1) / 2 + 1
        else:
            teams = teams / 2
            total += teams
    return total

def numberOfMatches(self, n):
    return n - 1

def numberOfMatches(self, n: int) -> int:
    ans = 0
    while n > 1:
        ans += (n // 2)
        n = (n // 2) + (n % 2)
    return ans

def numberOfMatches(self, n):  # рекурсия
    if n == 1:
        return 0

    if n % 2 == 0:
        numberOfMatchesInCurrentRound = n / 2
        numberOfTeamsForNextRound = n / 2
    else:
        numberOfMatchesInCurrentRound = (n - 1) / 2
        numberOfTeamsForNextRound = (n - 1) / 2 + 1

    return numberOfMatchesInCurrentRound + self.numberOfMatches(numberOfTeamsForNextRound)

def numberOfMatches(self, n):  # рекурсия
    if n == 1:
        return 0

    if n % 2 == 0:
        return n / 2 + self.numberOfMatches(n / 2)
    else:
        return (n - 1) / 2 + self.numberOfMatches((n - 1) / 2 + 1)

# =============================================================================
# 1588. Sum of All Odd Length Subarrays

def sumOddLengthSubarrays(self, arr: List[int]) -> int:
    summ = 0
    n = len(arr)
    for srez in range(1, n + 1, 2):
        for index in range(n - srez + 1):
            subarray = arr[index:index + srez]
            summ += sum(subarray)
    return summ

def sumOddLengthSubarrays(self, arr: List[int]) -> int:
    s = 0
    for i in range(len(arr)):
        for j in range(i, len(arr), 2):
            s += sum(arr[i:j + 1])
    return s

# =============================================================================
# 9. Palindrome Number

def isPalindrome(x):
    def isPalindrome(self, x):
        return str(x) == str(x)[::-1]

def isPalindrome(self, x):
    if x == 0:
        return True
    if x < 0 or x % 10 == 0:
        return False

    half = 0
    while half < x:
        half = (x % 10) + half * 10
        x = x // 10
    if half > x:
        half //= 10
    return half == x

def isPalindrome(self, x):
    if x < 0:
        return False
    rev = 0
    new_x = x
    while new_x > 0:
        rev = (new_x % 10) + rev * 10
        new_x = new_x // 10
    return rev == x

# =============================================================================
# 2810. Faulty Keyboard

def finalString(self, s):
    res = ''
    for i in range(len(s)):
        if s[i] == 'i':
            res = res[::-1]
        else:
            res += s[i]
    return res

# =============================================================================
# 2956. Find Common Elements Between Two Arrays

def findIntersectionValues(self, nums1, nums2):
    count1 = 0
    count2 = 0
    for i in range(len(nums1)):
        if nums1[i] in nums2:
            count1 += 1
    for i in range(len(nums2)):
        if nums2[i] in nums1:
            count2 += 1
    res = [count1, count2]
    return res

# =============================================================================
# 2656. Maximum Sum With Exactly K Elements

def maximizeSum(self, nums, k):
    maxx = max(nums)
    res = maxx
    for i in range(k - 1):
        maxx += 1
        res += maxx
    return res

def maximizeSum(self, nums: List[int], k: int) -> int:
    return k * max(nums) + k * (k - 1) // 2

# =============================================================================
# 1534. Count Good Triplets

def countGoodTriplets(self, arr, a, b, c):
    count = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if abs(arr[i] - arr[j]) <= a and abs(arr[j] - arr[k]) <= b and abs(arr[i] - arr[k]) <= c:
                    count += 1
    return count

def countGoodTriplets(self, arr, a, b, c):
    count = 0
    n = len(arr)
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            if abs(arr[i] - arr[j]) <= a:
                for k in range(j + 1, n):
                    if abs(arr[j] - arr[k]) <= b and abs(arr[i] - arr[k]) <= c:
                        count += 1
    return count

# =============================================================================
# 1844. Replace All Digits with Characters

def replaceDigits(self, s):
    def shift(letter, num):
        x = ord(
            letter)  ## ord()  ordinal(порядковый номер) возвращает юникодовый код символа. между большой и маленькой англ! буквой 32 символа
        return chr(x + num)

    res = ''
    for i in range(1, len(s), 2):
        y = shift(s[i - 1], int(s[i]))
        res += s[i - 1]
        res += y
    if len(s) % 2 != 0:
        res += s[-1]
    return res

# =============================================================================
# 1572. Matrix Diagonal Sum

def diagonalSum(self, mat):
    res = 0
    for i in range(len(mat)):
        res += mat[i][i]
        res += mat[i][-i - 1]
    if len(mat) % 2 != 0:
        x = len(mat) // 2
        res -= mat[x][x]
    return res

# =============================================================================
# 2315. Count Asterisks
# 1)
def countAsterisks(self, s):
    count = 0
    in_bars = False
    for i in range(len(s)):
        if s[i] == '|':
            in_bars = not in_bars
        if not in_bars:
            if s[i] == '*':
                count += 1
    return count

# 2)
def countAsterisks(self, s):
    count = 0
    new = s.split('|')
    for i in range(0, len(new), 2):
        for j in new[i]:
            if j == '*':
                count += 1
    return count

# =============================================================================
# 2913. Subarrays Distinct Element Sum of Squares I

def sumCounts(self, nums):
    res = 0
    for len_sub in range(1, len(nums) + 1):
        for index in range(len(nums) - len_sub + 1):
            sub = nums[index:index + len_sub]
            res += len(set(sub)) ** 2
    return res

# =============================================================================
# 1863. Sum of All Subset XOR Totals

from itertools import combinations

class Solution(object):
    def subsetXORSum(self, nums):
        res = 0
        for i in range(1, len(nums) + 1):  # длина
            for sub in combinations(nums, i):
                first_sub = 0
                for num in sub:
                    first_sub = first_sub ^ num
                res += first_sub
        return res

# =============================================================================
# 1436. Destination City

def destCity(self, paths):
    new1 = []
    new2 = []
    for i in paths:
        new1.append(i[0])
        new2.append(i[1])
    for i in new2:
        if i not in new1:
            return i

# =============================================================================
# 2485. Find the Pivot Integer

def pivotInteger(self, n):  # очень медленно
    if n == 1:
        return 1
    for i in range(2, n + 1):
        first = sum(list(range(1, i + 1)))
        sec = sum(list(range(i, n + 1)))
        if first == sec:
            return i
    return -1

def pivotInteger(self, n):  # в обратную сторону, тоже медленно
    if n == 1:
        return 1
    for i in range(n, 1, -1):
        first = sum(range(1, i + 1))
        sec = sum(range(i, n + 1))
        if first == sec:
            return i
    return -1

## через формулу арифметической прогрессии
def pivotInteger(self, n):
    temp = (n * n + n) // 2
    sq = int(math.sqrt(temp))
    if sq * sq == temp:
        return sq
    return -1

# =============================================================================
# 2000. Reverse Prefix of Word

def reversePrefix(self, word, ch):
    for i in range(len(word)):
        if word[i] == ch:
            new = word[:i + 1][::-1] + word[i + 1:]
            return new
    return word

# =============================================================================
# 1967. Number of Strings That Appear as Substrings in Word

def numOfStrings(self, patterns, word):
    count = 0
    for i in range(len(patterns)):
        if patterns[i] in word:
            count += 1
    return count

# =============================================================================
# 1351. Count Negative Numbers in a Sorted Matrix

def countNegatives(self, grid):
    count = 0
    for row in grid:
        for val in row:
            if val < 0:
                count += 1
    return count

def countNegatives(self, grid):
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] < 0:
                count += 1
    return count

# =============================================================================
# 1282. Group the People Given the Group Size They Belong To

def groupThePeople(self, groupSizes):
    dic = {}
    for i in range(len(groupSizes)):
        x = groupSizes[i]
        if x not in dic:
            dic[x] = []
        dic[x].append(i)
    res = []
    for key, value in dic.items():
        tmp = []
        for elem in value:
            tmp.append(elem)
            if len(tmp) == key:
                res.append(tmp)
                tmp = []
    return res

# =============================================================================
# 2778. Sum of Squares of Special Elements

def sumOfSquares(self, nums):
    res = 0
    n = len(nums)
    for i in range(1, n + 1):
        if n % i == 0:
            res += nums[i - 1] ** 2
    return res

def sumOfSquares(self, nums):
    res = 0
    n = len(nums)
    for i in range(n):
        if n % (i + 1) == 0:
            res += nums[i] ** 2
    return res

# =============================================================================
# 1309. Decrypt String from Alphabet to Integer Mapping

def freqAlphabets(self, s):
    alp = {'1': 'a',
           '2': 'b',
           '3': 'c',
           '4': 'd',
           '5': 'e',
           '6': 'f',
           '7': 'g',
           '8': 'h',
           '9': 'i',
           '10#': 'j',
           '11#': 'k',
           '12#': 'l',
           '13#': 'm',
           '14#': 'n',
           '15#': 'o',
           '16#': 'p',
           '17#': 'q',
           '18#': 'r',
           '19#': 's',
           '20#': 't',
           '21#': 'u',
           '22#': 'v',
           '23#': 'w',
           '24#': 'x',
           '25#': 'y',
           '26#': 'z'}
    res = ''
    i = 0
    while i < len(s):
        if i + 2 < len(s) and s[i + 2] == '#':
            res += alp[s[i:i + 3]]
            i += 3
        else:
            res += alp[s[i]]
            i += 1
    return res

# =============================================================================
# 2176. Count Equal and Divisible Pairs in an Array

def countPairs(self, nums, k):
    res = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j] and (i * j) % k == 0:
                res += 1
    return res

# =============================================================================
# 1768. Merge Strings Alternately

def mergeAlternately(self, word1, word2):
    res = ''
    maxx = max(len(word1), len(word2))
    for i in range(maxx):
        if i < len(word1):
            res += word1[i]
        if i < len(word2):
            res += word2[i]
    return res

def mergeAlternately(self, word1, word2):
    res = ''
    minn = min(len(word1), len(word2))
    for i in range(minn):
        res += word1[i]
        res += word2[i]
    if len(word1) > len(word2):
        res += word1[len(word2):]
    if len(word1) < len(word2):
        res += word2[len(word1):]
    return res

from itertools import zip_longest

def mergeAlternately(self, word1: str, word2: str) -> str:
    res = ''
    for i, j in zip_longest(word1, word2, fillvalue=''):
        res += i
        res += j
    return res

def mergeAlternately(self, word1: str, word2: str) -> str:
    res = ''
    maxx = max(len(word1), len(word2))
    for i in range(maxx):
        try:
            res += word1[i]
        except IndexError:
            pass
        try:
            res += word2[i]
        except IndexError:
            pass
    return res

# =============================================================================
# 3019. Number of Changing Keys

def countKeyChanges(self, s):
    s = s.lower()
    res = 0
    for i in range(1, len(s)):
        if s[i] != s[i - 1]:
            res += 1
    return res

# =============================================================================
# 1748. Sum of Unique Elements

def sumOfUnique(self, nums):
    uniq = []
    for i in range(len(nums)):
        if nums[i] not in nums[0:i] and nums[i] not in nums[i + 1:]:
            uniq.append(nums[i])
    return sum(uniq)

def sumOfUnique(self, nums):
    uniq = []
    for i in nums:
        if nums.count(i) == 1:
            uniq.append(i)
    return sum(uniq)

# =============================================================================
# 1475. Final Prices With a Special Discount in a Shop

def finalPrices(self, prices):
    for i in range(len(prices) - 1):
        for j in range(i + 1, len(prices)):
            if prices[i] >= prices[j]:
                prices[i] = prices[i] - prices[j]
                break
    return prices

prices = [10, 1, 1, 6]

def finalPrices(self, prices):
    for i in range(len(prices) - 1):
        j = i + 1
        while j < len(prices) and prices[i] < prices[j]:
            j += 1
        if j < len(prices) and prices[i] >= prices[j]:
            prices[i] = prices[i] - prices[j]
    return prices

# =============================================================================
# 1979. Find Greatest Common Divisor of Array

def findGCD(self, nums):
    maxx = max(nums)
    minn = min(nums)
    i = minn
    while i > 0:
        if maxx % i == 0 and minn % i == 0:
            return i
        else:
            i -= 1

def findGCD(self, nums):
    maxx = max(nums)
    minn = min(nums)
    for i in range(minn, 0, -1):  ## убывающий цикл, на уменьшение. цикл в обратную сторону
        if maxx % i == 0 and minn % i == 0:
            return i

# =============================================================================
# 1374. Generate a String With Characters That Have Odd Counts

def generateTheString(self, n):
    if n % 2 == 0:
        res = 'a' * (n - 1) + 'b'
    else:
        res = 'a' * n
    return res

# =============================================================================
# 2185. Counting Words With a Given Prefix

def prefixCount(self, words, pref):
    count = 0
    for word in words:
        if word[:len(pref)] == pref:
            count += 1
    return count

def prefixCount(self, words, pref):
    count = 0
    for word in words:
        if word.startswith(pref):  ## метод проверяет с чего начинается
            count += 1
    return count

# =============================================================================
# 2500. Delete Greatest Value in Each Row

def deleteGreatestValue(self, grid):
    count = 0
    while len(grid[0]) > 0:
        tmp = 0
        for i in range(len(grid)):
            if tmp < max(grid[i]):
                tmp = max(grid[i])
            grid[i].remove(max(grid[i]))  ## удалить первое найденное значение, pop() - удаление по индексу
        count += tmp
    return count

# =============================================================================
# 807. Max Increase to Keep City Skyline

def maxIncreaseKeepingSkyline(self, grid):
    res = 0
    n = len(grid)
    maxx = [0] * n

    for i in range(n):
        for j in range(n):
            if maxx[j] < grid[i][j]:
                maxx[j] = grid[i][j]

    for i in range(n):
        for j in range(n):
            minn = min(maxx[j], max(grid[i]))
            if grid[i][j] < minn:
                res = res + (minn - grid[i][j])
    return res

def maxIncreaseKeepingSkyline(self, grid):
    res = 0
    n = len(grid)
    max_col = [0] * n
    max_row = [0] * n

    for i in range(n):
        for j in range(n):
            max_col[j] = max(max_col[j], grid[i][j])
            max_row[i] = max(max_row[i], grid[i][j])

    for i in range(n):
        for j in range(n):
            minn = min(max_col[j], max_row[i])
            if grid[i][j] < minn:
                res = res + (minn - grid[i][j])
    return res

# =============================================================================
# 2215. Find the Difference of Two Arrays

def findDifference(self, nums1, nums2):
    res = [[], []]
    x = set(nums1)
    y = set(nums2)
    for i in x:
        if i not in y:
            res[0].append(i)
    for i in y:
        if i not in x:
            res[1].append(i)
    return res

# =============================================================================
# 1725. Number Of Rectangles That Can Form The Largest Square

def countGoodRectangles(self, rectangles):
    new = []
    for i in rectangles:
        x = min(i)
        new.append(x)
    y = max(new)
    res = new.count(y)
    return res

def countGoodRectangles(self, rectangles):
    new = []
    for i in rectangles:
        new.append(min(i))
    return new.count(max(new))

# =============================================================================
# 1295. Find Numbers with Even Number of Digits

def findNumbers(self, nums):
    count = 0
    for i in nums:
        if len(str(i)) % 2 == 0:
            count += 1
    return count

# =============================================================================
# 2032. Two Out of Three

def twoOutOfThree(self, nums1, nums2, nums3):
    res = []
    uniq = set(nums1 + nums2 + nums3)
    for i in uniq:
        tmp = 0
        if i in nums1:
            tmp += 1
        if i in nums2:
            tmp += 1
        if tmp > 1:
            res.append(i)
            continue
        if i in nums3:
            tmp += 1
        if tmp > 1:
            res.append(i)
    return res

# =============================================================================
# 2089. Find Target Indices After Sorting Array

def targetIndices(self, nums, target):
    res = []
    x = sorted(nums)
    for i in range(len(nums)):
        if x[i] == target:
            res.append(i)
    return res

# =============================================================================
# 2864. Maximum Odd Binary Number

def maximumOddBinaryNumber(self, s):
    x = ''.join(sorted(s, reverse=True))
    return x[1:] + x[0]

# =============================================================================
# 977. Squares of a Sorted Array

def sortedSquares(self, nums):
    res = []
    for i in nums:
        res.append(i ** 2)
    return sorted(res)

def sortedSquares(self, nums):  ## за один проход, но insert медленный
    res = []
    left = 0
    right = len(nums) - 1
    while left <= right:
        if nums[left] ** 2 > nums[right] ** 2:
            res.insert(0, nums[left] ** 2)
            left += 1
        else:
            res.insert(0, nums[right] ** 2)
            right -= 1
    return res

def sortedSquares(self, nums):
    res = []
    left = 0
    right = len(nums) - 1
    while left <= right:
        if nums[left] ** 2 > nums[right] ** 2:
            res.append(nums[left] ** 2)
            left += 1
        else:
            res.append(nums[right] ** 2)
            right -= 1
    res.reverse()
    return res

def sortedSquares(self, nums):
    res = [0] * len(nums)
    left = 0
    right = len(nums) - 1
    for position in range(len(nums) - 1, -1, -1):
        lft_sqr = nums[left] ** 2
        rght_sqr = nums[right] ** 2
        if lft_sqr > rght_sqr:
            res[position] = lft_sqr
            left += 1
        else:
            res[position] = rght_sqr
            right -= 1
    return res

# =============================================================================
# 1941. Check if All Characters Have Equal Number of Occurrences

def areOccurrencesEqual(self, s: str) -> bool:
    uniq = list(s)
    uniq = set(uniq)
    count = s.count(s[0])
    for i in uniq:
        if s.count(i) != count:
            return False
    return True

def areOccurrencesEqual(self, s: str) -> bool:
    dic = {}
    for i in s:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    return len(set(dic.values())) == 1

def areOccurrencesEqual(self, s: str) -> bool:
    count = []
    for i in set(s):
        count.append(s.count(i))
    return len(set(count)) == 1

# =============================================================================
# 1207. Unique Number of Occurrences

def uniqueOccurrences(self, arr: List[int]) -> bool:
    dic = {}
    for i in arr:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    return len(dic.values()) == len(set(dic.values()))

# =============================================================================
# 1750. Minimum Length of String After Deleting Similar Ends

def minimumLength(s: str) -> int:
    while len(s) > 1 and s[0] == s[-1]:
        sim = s[0]
        while len(s) > 0 and s[0] == sim:
            s = s[1:]
        while len(s) > 0 and s[-1] == sim:
            s = s[:-1]
    return len(s)

# =============================================================================
# 2733. Neither Minimum nor Maximum

def findNonMinOrMax(self, nums: List[int]) -> int:
    if len(nums) >= 2:
        nums.remove(max(nums))
        nums.remove(min(nums))
        if len(nums) > 0:
            return nums[0]
    return -1

def findNonMinOrMax(self, nums: List[int]) -> int:
    if len(nums) > 2:
        nums = sorted(nums)
        return nums[1]
        return -1

# =============================================================================
# 2951. Find the Peaks

def findPeaks(self, mountain: List[int]) -> List[int]:
    res = []
    for i in range(1, len(mountain) - 1):
        if mountain[i] > mountain[i - 1] and mountain[i] > mountain[i + 1]:
            res.append(i)
    return res

# =============================================================================
# 3005. Count Elements With Maximum Frequency

def maxFrequencyElements(self, nums: List[int]) -> int:
    dic = {}
    for i in nums:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    maxx = max(dic.values())
    count = 0
    for value in dic.values():
        if value == maxx:
            count += maxx
    return count

# =============================================================================
# 2540. Minimum Common Value

def getCommon(self, nums1: List[int], nums2: List[int]) -> int:
    y = set(nums2)
    for i in nums1:
        if i in y:
            return i
    return -1

# =============================================================================
# 349. Intersection of Two Arrays

def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
    res = []
    nums2 = set(nums2)
    for i in nums1:
        if i in nums2:
            res.append(i)
    return set(res)

# =============================================================================
# 791. Custom Sort String

def customSortString(order: str, s: str) -> str:
    res = ''
    for i in range(len(order)):
        while order[i] in s:
            res += order[i]
            s = s.replace(order[i], '', 1)
    return res + s

# =============================================================================
# 905. Sort Array By Parity

def sortArrayByParity(self, nums: List[int]) -> List[int]:
    res = []
    for i in nums:
        if i % 2 != 0:
            res.append(i)
        else:
            res.insert(0, i)
    return res

def sortArrayByParity(self, nums: List[int]) -> List[int]:
    even = []
    odd = []
    for i in nums:
        if i % 2 == 0:
            even.append(i)
        else:
            odd.append(i)
    return even + odd

# =============================================================================
# 1304. Find N Unique Integers Sum up to Zero

def sumZero(self, n: int) -> List[int]:
    res = []
    for i in range(1, n // 2 + 1):
        res.append(i)
        res.append(-i)
    if n % 2 != 0:
        res.append(0)
    return res

# =============================================================================
# 961. N-Repeated Element in Size 2N Array

def repeatedNTimes(self, nums: List[int]) -> int:
    dic = {}
    n = len(nums) // 2
    for i in nums:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    for key, val in dic.items():
        if val == n:
            return int(key)

# =============================================================================
# 2678. Number of Senior Citizens

def countSeniors(self, details: List[str]) -> int:
    count = 0
    for i in details:
        if int(i[-4:-2]) > 60:
            count += 1
    return count

# =============================================================================
# 525. Contiguous Array

def findMaxLength(nums: List[int]) -> int:
    class Solution:
        def findMaxLength(self, nums: List[int]) -> int:
            dic = {0: -1}
            summ = 0
            i = 0
            max_len = 0
            while i < len(nums):
                if nums[i] == 1:
                    summ += 1
                if nums[i] == 0:
                    summ -= 1
                if summ not in dic:
                    dic[summ] = i
                else:
                    starti = dic[summ]
                    lenght = i - starti
                    if lenght > max_len:
                        max_len = lenght
                i += 1
            return max_len

# =============================================================================
# 1051. Height Checker

def heightChecker(self, heights: List[int]) -> int:
    sort = sorted(heights)
    count = 0
    for i in range(len(heights)):
        if heights[i] != sort[i]:
            count += 1
    return count

# =============================================================================
# 2119. A Number After a Double Reversal

def isSameAfterReversals(self, num):
    new = int(str(num)[::-1])
    if int(str(new)[::-1]) == num:
        return True

# =============================================================================
# 2643. Row With Maximum Ones

def rowAndMaximumOnes(self, mat: List[List[int]]) -> List[int]:
    summ = []
    for i in mat:
        summ.append(sum(i))
    x = max(summ)
    return [summ.index(x), max(summ)]

# =============================================================================
# 2716. Minimize String Length

def minimizedStringLength(self, s: str) -> int:
    return len(set(s))

# =============================================================================
# 1450. Number of Students Doing Homework at a Given Time

# 1й вариант быстрее
def busyStudent(self, startTime: List[int], endTime: List[int], queryTime: int) -> int:
    count = 0
    for i in range(len(startTime)):
        if startTime[i] <= queryTime and queryTime <= endTime[i]:  # 1й вариант быстрее
            count += 1
    return count

def busyStudent(self, startTime: List[int], endTime: List[int], queryTime: int) -> int:
    count = 0
    for i in range(len(startTime)):
        if startTime[i] <= queryTime <= endTime[i]:
            count += 1
    return count

# ============================================================================
# 2341. Maximum Number of Pairs in Array

def numberOfPairs(self, nums: List[int]) -> List[int]:
    count = 0
    for i in range(len(nums)):
        x = nums[i]
        if x == None:
            continue
        if nums.count(x) >= 2:
            nums[i] = None
            index = nums.index(x)
            nums[index] = None
            count += 1
    while None in nums:
        nums.remove(None)  ## удаление элемента по значению
    return [count, len(nums)]

# =============================================================================
# 287. Find the Duplicate Number

def findDuplicate(self, nums: List[int]) -> int:
    dic = {}
    for i in nums:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    for key, val in dic.items():
        if val > 1:
            return key

def findDuplicate(self, nums: List[int]) -> int:
    for i in nums:
        if nums.count(i) > 1:
            return i  # очень медленно

# =============================================================================
# 876. Middle of the Linked List
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def middleNode(self, head: ListNode) -> ListNode:
    count = 0
    curr = head
    while curr != None:
        count += 1
        curr = curr.next
    count = count // 2 + 1

    tmp = 1
    curr = head
    while tmp < count:
        tmp += 1
        curr = curr.next
    return curr

# 2
def middleNode(self, head: ListNode) -> ListNode:
    step1 = head
    step2 = head
    while step2 != None and step2.next != None:
        step1 = step1.next
        step2 = step2.next
        if step2 != None:
            step2 = step2.next
    return step1

def middleNode(self, head: ListNode) -> ListNode:
    step1 = head
    step2 = head
    while step2 and step2.next:
        step1 = step1.next
        step2 = step2.next.next
    return step1

# =============================================================================
# 2965. Find Missing and Repeated Values

def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
    new = [0]
    for i in grid:
        new.extend(i)
    new.append(len(new))
    new = sorted(new)
    twice = None
    miss = None
    for i in range(len(new) - 1):
        if new[i + 1] - new[i] == 0:
            twice = new[i]
        elif new[i + 1] - new[i] == 2:
            miss = new[i] + 1
    return [twice, miss]

# =============================================================================
# 206. Reverse LinkedLinked List

def reverseList(self, head: ListNode) -> ListNode:
    curr = head
    prev = None
    while curr is not None:
        tail = curr.next
        curr.next = prev
        prev = curr
        curr = tail
    return prev

## recursion
def reverseList(self, head: ListNode) -> ListNode:
    if head is None:
        return None
    if head.next is None:
        return head
    tail = head.next
    head.next = None
    reversed_tail = self.reverseList(tail)
    tail.next = head
    return reversed_tail

# =============================================================================
#

def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:
    res = ListNode()
    head = res

    while list1 and list2:
        if list1.val <= list2.val:
            res.next = list1
            list1 = list1.next
        else:
            res.next = list2
            list2 = list2.next
        res = res.next

    res.next = list1 or list2
    return head.next

def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:
    curr1 = list1
    curr2 = list2

    if curr1 is None and curr2 is None:
        return None
    if curr1 is None:
        return curr2
    if curr2 is None:
        return curr1

    if list1.val <= list2.val:
        res = curr1
        curr1 = curr1.next
    else:
        res = curr2
        curr2 = list2.next
    head = res

    while curr1 and curr2:
        if curr1.val <= curr2.val:
            res.next = curr1
            res = curr1
            curr1 = curr1.next
        else:
            res.next = curr2
            res = curr2
            curr2 = curr2.next

    if curr1 is None:
        tmp = curr2
    elif curr2 is None:
        tmp = curr1
    res.next = tmp
    return head

# =============================================================================
# 2351. First Letter to Appear Twice

def repeatedCharacter(self, s: str) -> str:
    dic = {}
    for i in range(len(s)):
        if s[i] not in dic:
            dic[s[i]] = 0
        dic[s[i]] += 1
        if dic[s[i]] == 2:
            return s[i]

# =============================================================================
# 2586. Count the Number of Vowel Strings in Range

def vowelStrings(self, words: List[str], left: int, right: int) -> int:
    new = words[left:right + 1]
    vowels = ['a', 'e', 'i', 'o', 'u']
    count = 0
    for word in new:
        if word[0] in vowels and word[-1] in vowels:
            count += 1
    return count

def vowelStrings(self, words: List[str], left: int, right: int) -> int:
    vowels = 'aeiou'
    count = 0
    for i in range(left, right + 1):
        if words[i][0] in vowels and words[i][-1] in vowels:
            count += 1
    return count

def vowelStrings(self, words: List[str], left: int, right: int) -> int:
    new = words[left:right + 1]
    vowels = ['a', 'e', 'i', 'o', 'u']
    count = 0
    for word in new:
        tmp = 0
        for vowel in vowels:
            if word[0] == vowel:
                tmp += 1
            if word[-1] == vowel:
                tmp += 1
            if tmp == 2:
                break
        if tmp == 2:
            count += 1
        else:
            tmp = 0
    return count

# =============================================================================
# 79. Word Search

## нужно куча всего:
# доска  board
# занятые буквы used
# оставшаяся часть слова word
# текущее положение row, column,

def search_letter(i_row, i_column, board, word, used):
    if word == "":
        return True
    if len(word) == 1 and board[i_row][i_column] == word[0] and (i_row, i_column) not in used:
        return True
    if i_column < len(board[i_row]) - 1:  # вправо
        if board[i_row][i_column + 1] == word[0] and (i_row, i_column + 1) not in used:
            used.add((i_row, i_column + 1))
            found = search_letter(i_row, i_column + 1, board, word[1:], used)
            if found:
                return True
            used.remove((i_row, i_column + 1))

    if i_row < len(board) - 1:  # вниз
        if board[i_row + 1][i_column] == word[0] and (i_row + 1, i_column) not in used:
            used.add((i_row + 1, i_column))
            found = search_letter(i_row + 1, i_column, board, word[1:], used)
            if found:
                return True
            used.remove((i_row + 1, i_column))

    if i_column > 0:  # влево
        if board[i_row][i_column - 1] == word[0] and (i_row, i_column - 1) not in used:
            used.add((i_row, i_column - 1))
            found = search_letter(i_row, i_column - 1, board, word[1:], used)
            if found:
                return True
            used.remove((i_row, i_column - 1))

    if i_row > 0:  # вверх
        if board[i_row - 1][i_column] == word[0] and (i_row - 1, i_column) not in used:
            used.add((i_row - 1, i_column))
            found = search_letter(i_row - 1, i_column, board, word[1:], used)
            if found:
                return True
            used.remove((i_row - 1, i_column))

    return False

def exist(self, board: List[List[str]], word: str) -> bool:
    for row in range(len(board)):
        for column in range(len(board[row])):
            if search_letter(row, column, board, word[::-1], set()):
                return True
    return False

# =============================================================================
# 1614. Maximum Nesting Depth of the Parentheses

def maxDepth(self, s: str) -> int:
    count = 0
    left = 0
    right = 0
    for i in range(len(s)):
        if s[i] == ')':
            right += 1
        if s[i] == '(':
            left += 1
        depth = left - right
        if depth > count:
            count = depth
    return count

def maxDepth(self, s: str) -> int:
    count = 0
    for i in range(len(s)):
        if s[i] == ')':
            left = 0
            right = 0
            index = 0
            while index < i:
                if s[index] == '(':
                    left += 1
                elif s[index] == ')':
                    right += 1
                index += 1
            depth = left - right
            if depth > count:
                count = depth
    return count

# =============================================================================
# 1544. Make The String Great

def makeGood(self, s: str) -> str:
    i = 1
    s = list(s)
    while i < len(s):
        if ord(s[i - 1]) - ord(s[i]) == 32 or ord(s[i]) - ord(s[i - 1]) == 32:  # перевод буквы в код
            s.pop(i)
            s.pop(i - 1)
            i = 1
        else:
            i += 1
    return ''.join(s)

# =============================================================================
# 1249. Minimum Remove to Make Valid Parentheses

def minRemoveToMakeValid(self, s: str) -> str:
    stack = []
    i = 0
    s = list(s)
    while i < len(s):
        if s[i] == '(':
            stack.append(i)
        if s[i] == ')':
            if stack:
                stack.pop()
            else:
                s.pop(i)
                i -= 1
        i += 1
    while stack:
        i = stack.pop()
        s.pop(i)
    return ''.join(s)

def minRemoveToMakeValid(self, s: str) -> str:
    stack = []
    i = 0
    s = list(s)
    while i < len(s):
        if s[i] == '(':
            stack.append((s[i], i))
        if s[i] == ')':
            if stack:
                stack.pop()
            else:
                s.pop(i)
                i -= 1
        i += 1
    while stack:
        brackets, i = stack.pop()
        s.pop(i)
    return ''.join(s)

# =============================================================================
# 678. Valid Parenthesis String

def checkValidString(self, s: str) -> bool:
    stack = []
    star = []
    i = 0
    s = list(s)
    while i < len(s):
        if s[i] == '*':
            star.append(i)
        elif s[i] == '(':
            stack.append(i)
        elif s[i] == ')':
            if stack:
                stack.pop()
            else:
                if star:
                    star.pop()
                else:
                    return False
        i += 1
    while stack and star:
        for j in star:
            if stack[0] < j:
                stack.pop(0)  ## удадение по индексу
                star.remove(j)  ## удадение по значению
                break
        else:
            return False
    return len(stack) == 0

# =============================================================================
# 1700. Number of Students Unable to Eat Lunch

def countStudents(self, students: List[int], sandwiches: List[int]) -> int:
    while sandwiches and sandwiches[0] in students:
        if students[0] == sandwiches[0]:
            students.pop(0)
            sandwiches.pop(0)
        else:
            x = students.pop(0)
            students.append(x)
    return len(sandwiches)

def countStudents(self, students: List[int], sandwiches: List[int]) -> int:
    while sandwiches and sandwiches[0] in students:
        students.remove(sandwiches[0])
        sandwiches.pop(0)
    return len(sandwiches)

def countStudents(self, students: List[int], sandwiches: List[int]) -> int:
    count = 0
    i = 0
    while i < len(students):
        if students[i] == sandwiches[i]:
            students.pop(i)
            sandwiches.pop(i)
            i -= 1
        elif students[i] != sandwiches[i] and sandwiches[i] in students:
            x = students.pop(i)
            students.append(x)
            i -= 1
        elif sandwiches[i] not in students:
            return len(sandwiches)
        i += 1
    return 0

# =============================================================================
# 2073. Time Needed to Buy Tickets

def timeRequiredToBuy(self, tickets: List[int], k: int) -> int:
    res = 0
    i = 0
    while tickets[k] > 0 and i < len(tickets):
        if tickets[i] != 0:
            tickets[i] = tickets[i] - 1
            res += 1
        i += 1
        if i == len(tickets):
            i = 0
    return res

# =============================================================================
# 950. Reveal Cards In Increasing Order

def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
    new = sorted(deck)
    res = []
    while new:
        res.insert(0, new[-1])
        new.pop(-1)
        if new:
            x = res.pop(-1)
            res.insert(0, x)
    return res

# =============================================================================
# 402. Remove K Digits

def removeKdigits(self, num: str, k: int) -> str:
    stack = []
    s = []  # new int num
    for i in num:
        x = int(i)
        s.append(x)

    for elem in num:
        if not stack:
            stack.append(elem)
            continue
        while stack and stack[-1] > elem and k > 0:
            stack.pop()
            k -= 1
        stack.append(elem)

    while k > 0:
        stack.pop()
        k -= 1

    res = ''
    for i in stack:
        res += str(i)
    res = res.lstrip("0")
    if res == '':
        return '0'
    return res

# =============================================================================
# 2255. Count Prefixes of a Given String

def countPrefixes(self, words: List[str], s: str) -> int:
    count = 0
    for i in words:
        x = len(i)
        if i == s[:x]:
            count += 1
    return count

def countPrefixes(self, words: List[str], s: str) -> int:
    count = 0
    for i in words:
        x = len(i)
        if s.startswith(i):  # начинается с
            count += 1
    return count

# =============================================================================
# 42. Trapping Rain Water

def trap(self, height: List[int]) -> int:
    water = 0
    for i in range(1, len(height) - 1):
        max_left = max(height[:i])
        max_right = max(height[i + 1:])
        x = min(max_left, max_right) - height[i]
        if x > 0:
            water += x
    return water

# =============================================================================
# 1935. Maximum Number of Words You Can Type

def canBeTypedWords(self, text: str, brokenLetters: str) -> int:
    count = 0
    text = text.split()
    for word in text:
        nobroken = True
        for letter in brokenLetters:
            if letter in word:
                nobroken = False
                break
        if nobroken:
            count += 1
    return count

def canBeTypedWords(self, text: str, brokenLetters: str) -> int:
    text = text.split()
    count = len(text)
    for word in text:
        for letter in brokenLetters:
            if letter in word:
                count -= 1
                break
    return count

# =============================================================================
# 2278. Percentage of Letter in String

def percentageLetter(self, s: str, letter: str) -> int:
    count = 0
    for i in s:
        if i == letter:
            count += 1
    return int(count / len(s) * 100)

# =============================================================================
# 1. Two Sum

def twoSum(self, nums: List[int], target: int) -> List[int]:
    for i in range(len(nums) - 1):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]

# =============================================================================
# 26. Remove Duplicates from Sorted Array

def removeDuplicates(self, nums):
    i = 0
    while i < len(nums):
        if nums.count(nums[i]) > 1:
            nums.pop(i)
        else:
            i += 1
    return len(nums)

# =============================================================================
# 28. Find the Index of the First Occurrence in a String

def strStr(self, haystack: str, needle: str) -> int:
    if needle in haystack:
        return haystack.index(needle)
    return - 1

# =============================================================================
# 14. Longest Common Prefix

def longestCommonPrefix(self, strs: List[str]) -> str:
    res = ''
    for j in range(len(strs[0])):  ## для каждого индекса буквы
        prefix = strs[0][:j + 1]
        for word in strs:  ## в каждом слове
            if not word.startswith(prefix):
                return res
        res = prefix
    return res

def longestCommonPrefix(self, strs: List[str]) -> str:
    for j in range(len(strs[0])):  ## для каждого индекса буквы
        for word in strs:  ## в каждом слове
            if j >= len(word) or word[j] != strs[0][j]:
                return strs[0][:j]
    return strs[0]

# =============================================================================
# 268. Missing Number

def missingNumber(self, nums: List[int]) -> int:
    for i in range(len(nums) + 1):
        if i not in nums:
            return i

# =============================================================================
# 136. Single Number

def singleNumber(self, nums: List[int]) -> int:
    dic = {}
    for i in nums:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    for key, values in dic.items():
        if values == 1:
            return key

# =============================================================================
# 3110. Score of a String

def scoreOfString(self, s: str) -> int:
    res = 0
    for i in range(len(s) - 1):
        res += abs(ord(s[i]) - ord(s[i + 1]))
    return res

# =============================================================================
# 2057. Smallest Index With Equal Value

def smallestEqual(self, nums: List[int]) -> int:
    for i in range(len(nums)):
        if i % 10 == nums[i]:  ## "Mod" (или "modulus") - остаток от деления одного числа на другое - %
            return i
    return -1

# =============================================================================
# 2529. Maximum Count of Positive Integer and Negative Integer

def maximumCount(self, nums: List[int]) -> int:
    pos = 0
    neg = 0
    for i in nums:
        if i < 0:
            neg += 1
        elif i > 0:
            pos += 1
        return max(pos, neg)

# =============================================================================
# 2785. Sort Vowels in a String

def sortVowels(self, s: str) -> str:
    to_sort = []
    indexes = []
    vowels = 'aeiouAEIOU'
    for i in range(len(s)):
        if s[i] in vowels:
            to_sort.append(ord(s[i]))
            indexes.append(i)
    if len(to_sort) > 0:
        to_sort = sorted(to_sort)
        indexes = sorted(indexes)
    else:
        return s
    new = []
    for i in s:
        new.append(i)

    for i in range(len(to_sort)):
        new.pop(indexes[i])
        new.insert(indexes[i], chr(to_sort[i]))
    return ''.join(new)

# =============================================================================
# 1021. Remove Outermost Parentheses

def removeOuterParentheses(self, s: str) -> str:
    stack = []
    res = ''
    group = ''
    for i in s:
        if i == '(':
            stack.append(i)
            group += i
        else:
            stack.pop()
            group += i
            if not stack:
                res += group[1:-1]
                group = ''
    return res

# =============================================================================
# 160. Intersection of Two Linked Lists

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

def lenList(self, head):
    count = 0
    curr = head
    while curr is not None:
        count += 1
        curr = curr.next
    return count

def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
    len1 = self.lenList(headA)
    len2 = self.lenList(headB)
    if len1 < len2:
        minn = headA
        maxx = headB
    else:
        minn = headB
        maxx = headA

    count = 0
    while count < (abs(len1 - len2)):
        maxx = maxx.next
        count += 1

    curr_head1 = maxx
    curr_head2 = minn
    while curr_head1 != None:
        if curr_head1 != curr_head2:
            curr_head1 = curr_head1.next
            curr_head2 = curr_head2.next
        else:
            return curr_head1
    return None

## draft
def lenList(self, head):
    count = 0
    curr = head
    while curr is not None:
        count += 1
        curr = curr.next
    return count

def copyList(self, head):
    curr = head
    prev = None
    while curr is not None:
        new_node = ListNode(curr.val)
        if prev != None:
            prev.next = new_node
        else:
            new_head = new_node
        prev = new_node
        curr = curr.next
    return new_head

def reverseList(self, head):
    curr = head
    prev = None
    while curr is not None:
        tail = curr.next
        curr.next = prev
        prev = curr
        curr = tail
    return prev

def old():
    rev_head1 = self.reverseList(self.copyList(headA))
    rev_head2 = self.reverseList(self.copyList(headB))

    curr_head1 = rev_head1
    curr_head2 = rev_head2
    while curr_head1 != None or curr_head2 != None:
        if curr_head1.val == curr_head2.val:
            curr_head1 = curr_head1.next
            curr_head2 = curr_head2.next
        else:
            return self.reverseList(curr_head1)
    return None

# =============================================================================
# 1710. Maximum Units on a Truck

def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
    count = 0
    boxTypes = sorted(boxTypes, key=lambda x: x[1],
                      reverse=True)  ## сортировка по 2му элементу (индексу 1 листа в листе)

    for box, items in boxTypes:
        if box <= truckSize:
            count += items * box
            truckSize -= box
        elif truckSize > 0:
            count += truckSize * items
            break
    return count

# =============================================================================
# 2496. Maximum Value of a String in an Array

def maximumValue(self, strs: List[str]) -> int:
    res = 0
    for i in strs:
        if i.isdigit():  ## проверяется, является ли строка целым числом (стринг = число?)
            i = int(i)
            if i > res:
                res = i
        else:
            if len(i) > res:
                res = len(i)
    return res

def maximumValue(self, strs: List[str]) -> int:
    res = 0
    for i in strs:
        if i.isdigit():
            res = max(int(i), res)
        else:
            res = max(len(i), res)
    return res

# =============================================================================
# 881. Boats to Save People

def numRescueBoats(self, people: List[int], limit: int) -> int:
    people = sorted(people, reverse=True)
    left = 0
    right = len(people) - 1
    boats = 0
    while left <= right:
        if people[left] + people[right] <= limit:
            right -= 1
        left += 1
        boats += 1
    return boats

def numRescueBoats(self, people: List[int], limit: int) -> int:
    people = sorted(people, reverse=True)
    weight = 0
    boats = 0
    while len(people) > 0:
        if people[0] == limit:
            boats += 1
            people.pop(0)
        elif people[0] < limit:  # если может уместиться 2е
            weight += people[0]
            people.pop(0)
            second = limit - weight
            if len(people) > 0 and people[-1] <= second:
                people.pop()
            weight = 0
            boats += 1
    return boats

# =============================================================================
# 237. Delete Node in a Linked List

def deleteNode(self, node):
    """
    :type node: ListNode
    :rtype: void Do not return anything, modify node in-place instead.
    """
    curr = node
    prev = None
    while curr.next != None:
        curr.val = curr.next.val
        prev = curr
        curr = curr.next
    prev.next = None

def deleteNode(self, node):
    """
    :type node: ListNode
    :rtype: void Do not return anything, modify node in-place instead.
    """
    curr = node
    while curr.next.next != None:
        curr.val = curr.next.val
        curr = curr.next
    curr.val = curr.next.val
    curr.next = None

# =============================================================================
# 1002. Find Common Characters

def dic(self, word):
    dic = {}
    for i in word:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    return dic

def commonChars(self, words: List[str]) -> List[str]:
    dics = []
    for word in words:
        dics.append(self.dic(word))

    res = dics[0]

    for i in dics[1:]:
        for key in res:
            res[key] = min(i.get(key, 0), res[key])

    result = []
    for key, value in res.items():
        for i in range(value):
            result.append(key)

    return result

# =============================================================================
# 500. Keyboard Row

def check(self, word, row):
    for i in word:
        if i.lower() not in row:
            return False
    return True

def findWords(self, words: List[str]) -> List[str]:
    row1 = set("qwertyuiop")
    row2 = set("asdfghjkl")
    row3 = set("zxcvbnm")
    res = []
    for word in words:
        if word[0].lower() in row1:
            if self.check(word, row1):
                res.append(word)
        elif word[0].lower() in row2:
            if self.check(word, row2):
                res.append(word)
        elif word[0].lower() in row3:
            if self.check(word, row3):
                res.append(word)
    return res

# =============================================================================
# 657. Robot Return to Origin

def judgeCircle(self, moves: str) -> bool:
    cur = [0, 0]
    for i in moves:
        if i == "R":
            cur[0] += 1
        elif i == "L":
            cur[0] -= 1
        elif i == "U":
            cur[1] += 1
        elif i == "D":
            cur[1] -= 1
    return cur == [0, 0]

# =============================================================================
# 2475. Number of Unequal Triplets in Array

def unequalTriplets(self, nums: List[int]) -> int:
    count = 0
    for i in range(len(nums) - 2):
        for j in range(i, len(nums) - 1):
            for k in range(j, len(nums)):
                if nums[i] != nums[j] and nums[i] != nums[k] and nums[j] != nums[k]:
                    count += 1
    return count

# =============================================================================
# 1299. Replace Elements with Greatest Element on Right Side

def replaceElements(self, arr: List[int]) -> List[int]:
    maxx = -1
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] > maxx:
            tmp = arr[i]
            arr[i] = maxx
            maxx = tmp
        else:
            arr[i] = maxx
    return arr

# =============================================================================
# 506. Relative Ranks

def findRelativeRanks(self, score: List[int]) -> List[str]:
    new = sorted(score, reverse=True)
    for i in range(len(score)):
        if score[i] == new[0]:
            score[i] = "Gold Medal"
        elif score[i] == new[1]:
            score[i] = "Silver Medal"
        elif score[i] == new[2]:
            score[i] = "Bronze Medal"
        else:
            score[i] = str(new.index(score[i]) + 1)
    return score

# =============================================================================
# 2085. Count Common Words With One Occurrence

def dic(self, words):
    dic = {}
    for i in words:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1

    remove_keys = []

    for key, val in dic.items():
        if val > 1:
            remove_keys.append(key)

    while remove_keys:
        key = remove_keys.pop()
        dic.pop(key)

    return dic

def countWords(self, words1: List[str], words2: List[str]) -> int:
    dic1 = self.dic(words1)
    dic2 = self.dic(words2)

    count = 0
    for key in dic2:
        if key in dic1:
            count += 1
    return count

def dic(self, words):
    dic = {}
    for i in words:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1

    uniq_word = []

    for key, val in dic.items():
        if val == 1:
            uniq_word.append(key)
    return uniq_word

def countWords(self, words1: List[str], words2: List[str]) -> int:
    uniq_word1 = self.dic(words1)
    uniq_word2 = self.dic(words2)

    count = 0
    for i in uniq_word2:
        if i in uniq_word1:
            count += 1
    return count

# =============================================================================
# 3075. Maximize Happiness of Selected Children

def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
    count = 0
    happiness = sorted(happiness, reverse=True)
    tmp = 0

    for i in range(k):
        if happiness[i] - tmp > 0:
            count += happiness[i] - tmp
            tmp += 1
        else:
            break

    return count

# =============================================================================
# 786. K-th Smallest Prime Fraction

def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
    new = []
    for i in range(len(arr) - 1):
        for j in range(i, len(arr)):
            x = arr[i] / arr[j]
            if x == 1:
                continue
            new.append([x, arr[i], arr[j]])
    new = sorted(new)
    return [new[k - 1][1], new[k - 1][2]]

# =============================================================================
# 1636. Sort Array by Increasing Frequency

def frequencySort(self, nums: List[int]) -> List[int]:
    def dic(nums):
        dic = {}
        for i in nums:
            if i not in dic:
                dic[i] = 0
            dic[i] += 1
        return dic

    dic_nums = dic(nums)
    res = []
    pair = []

    for key, freq in dic_nums.items():
        pair.append([freq, key])

    pair = sorted(pair, key=lambda x: x[1], reverse=True)
    pair = sorted(pair, key=lambda x: x[0])

    for i in pair:
        res.extend([i[1]] * i[0])
    return res

def dic(self, nums):
    dic = {}
    for i in nums:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    return dic

def frequencySort(self, nums: List[int]) -> List[int]:
    dic_nums = self.dic(nums)

    nums.sort(reverse=True)
    nums.sort(key=lambda x: dic_nums[x])

    return nums

# =============================================================================
# 1637. Widest Vertical Area Between Two Points Containing No Points

def maxWidthOfVerticalArea(self, points: List[List[int]]) -> int:
    points = sorted(points, key=lambda x: x[0])  # сортировка по 1му элементу листа в листе
    widest = 0
    for i in range(len(points) - 1):
        diff = points[i + 1][0] - points[i][0]
        if diff > widest:
            widest = diff
    return widest

def maxWidthOfVerticalArea(self, points: List[List[int]]) -> int:
    points.sort()
    widest = 0
    for i in range(len(points) - 1):
        diff = points[i + 1][0] - points[i][0]
        if diff > widest:
            widest = diff
    return widest

# =============================================================================
# 2373. Largest Local Values in a Matrix

def largestLocal(self, grid: List[List[int]]) -> List[List[int]]:
    length = len(grid) - 1
    res = []
    for row in range(1, length):
        rows_max = []
        for column in range(1, length):
            maxx = max(grid[row][column], grid[row][column + 1], grid[row][column - 1], grid[row - 1][column],
                       grid[row - 1][column + 1], grid[row - 1][column - 1], grid[row + 1][column],
                       grid[row + 1][column + 1], grid[row + 1][column - 1])
            rows_max.append(maxx)
        res.append(rows_max)
    return res

# =============================================================================
# 3065. Minimum Operations to Exceed Threshold Value I

def minOperations(self, nums: List[int], k: int) -> int:
    nums = sorted(nums)
    steps = 0
    for i in range(len(nums)):
        if nums[i] < k:
            steps += 1
        else:
            break
    return steps

def minOperations(self, nums: List[int], k: int) -> int:
    steps = 0
    for i in range(len(nums)):
        if nums[i] < k:
            steps += 1
    return steps

def minOperations(self, nums: List[int], k: int) -> int:
    nums = sorted(nums)
    if k in nums:
        i = nums.index(k)
        return len(nums[:i])
    else:
        steps = 0
        for i in range(len(nums)):
            if nums[i] < k:
                steps += 1
            else:
                break
        return steps

# =============================================================================
# 3131. Find the Integer Added to Array I

def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
    nums1 = sorted(nums1)
    nums2 = sorted(nums2)
    return nums2[0] - nums1[0]

def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
    return min(nums2) - min(nums1)

def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
    return (sum(nums2) - sum(nums1)) // len(nums1)

# =============================================================================
# 2037. Minimum Number of Moves to Seat Everyone

def minMovesToSeat(self, seats: List[int], students: List[int]) -> int:
    seats = sorted(seats)
    students = sorted(students)
    count = 0
    for i in range(len(seats)):
        if seats[i] == students[i]:
            continue
        else:
            count += abs(seats[i] - students[i])
    return count

def minMovesToSeat(self, seats: List[int], students: List[int]) -> int:
    seats = sorted(seats)
    students = sorted(students)
    count = 0
    for i in range(len(seats)):
        count += abs(seats[i] - students[i])
    return count

# =============================================================================
# 2108. Find First Palindromic String in the Array

def firstPalindrome(self, words: List[str]) -> str:
    res = ''
    for i in words:
        if i == i[::-1]:
            return i
    return res

# =============================================================================
# 2960. Count Tested Devices After Test Operations

def countTestedDevices(self, batteryPercentages: List[int]) -> int:
    res = 0
    step = 0
    for i in batteryPercentages:
        if i - step > 0:
            res += 1
            step += 1
    return res

# =============================================================================
# 1827. Minimum Operations to Make the Array Increasing

def minOperations(self, nums: List[int]) -> int:
    count = 0
    tmp = nums[0]
    for i in range(1, len(nums)):
        if tmp >= nums[i]:
            x = tmp - nums[i] + 1
            tmp += 1
            count += x
        else:
            tmp = nums[i]
    return count

# =============================================================================
# 66. Plus One

def plusOne(self, digits: List[int]) -> List[int]:
    for i in range(len(digits) - 1, -1, -1):
        if digits[i] == 9:
            digits[i] = 0
        else:
            digits[i] += 1
            return digits
    return [1] + digits

def plusOne(self, digits: List[int]) -> List[int]:
    new = ''
    for i in digits:
        new += str(i)
    new = int(new) + 1
    new = str(new)
    res = []
    for i in new:
        res.append(int(i))
    return res

# =============================================================================
# 78. Subsets

def subsets(self, nums: List[int]) -> List[List[int]]:
    res = []
    for i in range(len(nums) + 1):  # длина
        for sub in combinations(nums, i):
            res.append(sub)
    return res

# =============================================================================
# 2706. Buy Two Chocolates

def buyChoco(self, prices: List[int], money: int) -> int:
    prices = sorted(prices)
    summ = prices[0] + prices[1]
    if summ > money:
        return money
    else:
        return money - summ

# =============================================================================
# 1046. Last Stone Weight

def lastStoneWeight(self, stones: List[int]) -> int:
    stones = sorted(stones, reverse=True)
    while len(stones) > 2:
        if stones[0] - stones[1] > 0:
            stones[0] -= stones[1]
            stones.pop(1)
            stones = sorted(stones, reverse=True)
        elif stones[0] - stones[1] == 0:
            stones.pop(1)
            stones.pop(0)
    if len(stones) == 2:
        return abs(stones[0] - stones[1])
    if len(stones) == 1:
        return stones[0]
    return 0

# =============================================================================
# 509. Fibonacci Number (recursion)

def fib(self, n: int) -> int:
    if n == 0:
        return 0
    if n == 1:
        return 1
    return self.fib(n - 1) + self.fib(n - 2)

@cache  # декоратор, сохраняет в словарь все посчитанные значения (работает намного быстрее, чем без нее)
def fib(self, n: int) -> int:
    if n == 0:
        return 0
    if n == 1:
        return 1
    return self.fib(n - 1) + self.fib(n - 2)

# =============================================================================
# 3136. Valid Word

def isValid(self, word: str) -> bool:
    vowel = 'aeiou'
    consonant = 'bcdfghjklmnpqrstvwxyz'
    digits = '0123456789'
    vow = False
    cons = False
    dig = False
    if len(word) < 3:
        return False

    word = word.lower()
    for i in word:
        if i in vowel:
            vow = True
        elif i in consonant:
            cons = True
        elif i in digits:
            dig = True
        else:
            return False
    return vow and cons

# =============================================================================
# 434. Number of Segments in a String

def countSegments(self, s: str) -> int:
    return len(s.split())

def countSegments(self, s: str) -> int:
    count = 0
    letter = False
    for i in s:
        if i == " " and letter:
            count += 1
            letter = False
        elif i != " ":
            letter = True
    if letter:
        count += 1
    return count

def countSegments(self, s: str) -> int:
    count = 0
    if len(s) == 0:
        return 0
    elif s[-1] != " ":
        count += 1

    stack = []
    for i in s:
        if i == " " and stack:
            count += 1
            stack = []
        elif i != " ":
            stack.append(i)
    return count

# =============================================================================
# 1608. Special Array With X Elements Greater Than or Equal X

def specialArray(self, nums: List[int]) -> int:
    nums = sorted(nums)
    count = 0
    minn = min(nums)
    maxx = max(nums)
    for i in range(1, maxx + 1):
        for j in nums:
            if j >= i:
                count += 1
        if count == i:
            return count
        else:
            count = 0
    return -1

# =============================================================================
# 2423. Remove Letter To Equalize Frequency

def counter(self, word):
    dic = {}
    for i in word:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    return dic

def check(self, dic):
    uniq = None
    for key, val in dic.items():
        if uniq is None:
            uniq = val
        else:
            if uniq != val:
                return False
    return True

def equalFrequency(self, word: str) -> bool:
    for i in range(len(word)):
        new = word[:i] + word[i + 1:]
        if self.check(self.counter(new)):
            return True
    return False

# =============================================================================
# 2591. Distribute Money to Maximum Children

def distMoney(self, money: int, children: int) -> int:
    child = [0] * children

    if money < children:
        return -1

    for i in range(children):  # выдали по 1
        if money > 0:
            money -= 1
            child[i] += 1

    for i in range(children):  # выдали по 1
        if money >= 7:
            money -= 7
            child[i] += 7

    if money > 0:
        for i in range(children - 1, -1, -1):
            if child[i] + money != 4:
                child[i] += money
                money = 0
            else:
                child[i] += money - 1
                money = 1
    return child.count(8)

# =============================================================================
# 1909. Remove One Element to Make the Array Strictly Increasing

def canBeIncreasing(self, nums: List[int]) -> bool:
    for i in range(len(nums)):
        new = nums[:i] + nums[i + 1:]
        increasing = True
        for j in range(1, len(new)):
            if new[j - 1] >= new[j]:
                increasing = False
                break
        if increasing:
            return True
    return increasing

# =============================================================================
# 2486. Append Characters to String to Make Subsequence

def appendCharacters(self, s: str, t: str) -> int:
    i_s = 0
    i_t = 0
    while i_s < len(s) and i_t < len(t):
        if s[i_s] == t[i_t]:
            i_t += 1
        i_s += 1
    return len(t[i_t:])

# =============================================================================
# 409. Longest Palindrome

def count_letters(self, word):
    dic = {}
    for i in word:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    return dic

def longestPalindrome(self, s: str) -> int:
    res = 0
    odd = False
    dic = self.count_letters(s)
    for key, val in dic.items():
        if val % 2 != 0:
            odd = True
            res -= 1
        res += val
    if odd:
        res += 1
    return res

# =============================================================================
# 846. Hand of Straights

def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
    if len(hand) % groupSize != 0:
        return False
    hand.sort()
    i = 0
    group = []
    while i < len(hand):
        if len(group) == 0:
            group.append(hand.pop(i))
        if i < len(hand) and len(group) < groupSize:
            if hand[i] == group[-1]:
                if len(hand) == 1:
                    return False
                i += 1
            elif hand[i] - group[-1] == 1:
                group.append(hand.pop(i))
            elif hand[i] - group[-1] > 1:
                return False
        if len(group) == groupSize:
            group = []
            i = 0
    if i == len(hand) and group:
        return False
    return True

# =============================================================================
# 648. Replace Words

def replaceWords(self, dictionary: List[str], sentence: str) -> str:
    new = sentence.split()
    for root in dictionary:
        for word in new:
            if word.startswith(root):
                new[new.index(word)] = root
    return ' '.join(new)

# =============================================================================
# 605. Can Place Flowers

def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
    count = 0
    for i in range(len(flowerbed)):
        if flowerbed[i] == 0:
            empty_left = (i == 0) or (flowerbed[i - 1] == 0)
            empty_right = (i == len(flowerbed) - 1) or (flowerbed[i + 1] == 0)

            if empty_left and empty_right:
                flowerbed[i] = 1
                count += 1
                if count >= n:
                    return True
    return count >= n

# =============================================================================
# 925. Long Pressed Name

def isLongPressedName(self, name: str, typed: str) -> bool:
    if name[0] != typed[0]:
        return False

    i = 0
    for j in typed:
        if i < len(name) and name[i] == j:
            i += 1
        elif name[i - 1] == j:
            continue
        else:
            return False
    return i == len(name)

# =============================================================================
# 1122. Relative Sort Array

def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
    arr1.sort()
    arr1.sort(key=lambda x: arr2.index(x) if x in arr2 else 1000)
    return arr1

# =============================================================================
# 75. Sort Colors

def sortColors(self, nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    nums.sort()

# =============================================================================
# 945. Minimum Increment to Make Array Unique

def minIncrementForUnique(self, nums: List[int]) -> int:
    count = 0
    nums.sort()
    for i in range(1, len(nums)):
        if nums[i] <= nums[i - 1]:
            diff = (nums[i - 1] - nums[i])
            nums[i] = nums[i] + diff + 1
            count = count + diff + 1
    return count

# =============================================================================
# 414. Third Maximum Number

def thirdMax(self, nums: List[int]) -> int:
    new = list(set(nums))
    new.sort(reverse=True)
    if len(new) > 2:
        return new[2]
    return new[0]

# =============================================================================
# 1346. Check If N and Its Double Exist

def checkIfExist(self, arr: List[int]) -> bool:
    if arr.count(0) == 1:
        arr.remove(0)
    for i in arr:
        if i * 2 in arr:
            return True
    return False

# =============================================================================
# 2047. Number of Valid Words in a Sentence

def countValidWords(self, sentence: str) -> int:
    count = 0
    words = sentence.split()
    digits = '0123456789'

    for word in words:
        valid = False
        for letter in word:
            if letter in digits:
                valid = False
                break
            elif letter == "-":
                if word.count(letter) > 1:
                    valid = False
                    break
                elif letter not in word[1:-1]:
                    valid = False
                    break
            elif letter == "!" or letter == "." or letter == ",":
                if letter in word[:-1]:
                    valid = False
                    break
                elif len(word) > 1 and word[-2] == "-":
                    valid = False
                    break
                else:
                    valid = True
            else:
                valid = True

        if valid:
            count += 1
    return count

# =============================================================================
# 263. Ugly Number

def isUgly(self, n: int) -> bool:
    while n > 1:
        if n % 2 == 0:
            n = n / 2
        elif n % 3 == 0:
            n = n / 3
        elif n % 5 == 0:
            n = n / 5
        else:
            break
    return n == 1

# =============================================================================
# 2259. Remove Digit From Number to Maximize Result

def removeDigit(self, number: str, digit: str) -> str:
    new = []
    for i in range(len(number)):
        if number[i] == digit:
            new.append(int(number[:i] + number[i + 1:]))
            number[:i] + number[i + 1:]
    return str(max(new))

# =============================================================================
# 2553. Separate the Digits in an Array

def separateDigits(self, nums: List[int]) -> List[int]:
    res = []
    for i in nums:
        num = str(i)
        for j in num:
            res.append(int(j))
    return res

# =============================================================================
# 1550. Three Consecutive Odds

def threeConsecutiveOdds(self, arr: List[int]) -> bool:
    count = 0
    for i in arr:
        if i % 2 != 0:
            count += 1
            if count == 3:
                return True
        else:
            count = 0
    return False

def threeConsecutiveOdds(self, arr: List[int]) -> bool:
    count = 0
    for i in arr:
        if i % 2 != 0:
            count += 1
        else:
            count = 0
        if count == 3:
            return True
    return False

# =============================================================================
# 350. Intersection of Two Arrays II

def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
    res = []
    nums1.sort()
    nums2.sort()
    i = 0
    j = 0
    while i < len(nums1) and j < len(nums2):
        if nums1[i] == nums2[j]:
            res.append(nums1[i])
            i += 1
            j += 1
        elif nums1[i] < nums2[j]:
            i += 1
        elif nums1[i] > nums2[j]:
            j += 1
    return res

# =============================================================================
# 1509. Minimum Difference Between Largest and Smallest Value in Three Moves

def minDifference(self, nums: List[int]) -> int:
    if len(nums) < 5:
        return 0

    nums.sort()
    min_diff = float("Inf")
    # inf = float("Inf") Бесконечность (Infinity),
    # neg_inf = float("-Inf") - Отрицательная бесконечность (-Infinity),
    # nan = float("NaN") - Не число (NaN - Not a Number)

    for i in range(4):
        min_diff = min(min_diff, nums[i - 4] - nums[i])
    return min_diff

def minDifference(self, nums: List[int]) -> int:
    if len(nums) < 5:
        return 0

    nums.sort()
    min_diff = 10000000000
    for i in range(4):
        min_diff = min(min_diff, nums[i - 4] - nums[i])
    return min_diff

def minDifference(self, nums: List[int]) -> int:
    if len(nums) < 5:
        return 0

    nums.sort()
    return min(
        nums[-4] - nums[0],
        nums[-3] - nums[1],
        nums[-2] - nums[2],
        nums[-1] - nums[3]
    )

# =============================================================================
# 2582. Pass the Pillow

def passThePillow(self, n: int, time: int) -> int:
    i = 0
    while i < n and time > 0:
        i += 1
        time -= 1
    if i == n - 1:
        while i > 0 and time > 0:
            i -= 1
            time -= 1
    return i + 1

# =============================================================================
# 1518. Water Bottles

def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
    drink = numBottles

    while numBottles // numExchange > 0:
        empty = numBottles // numExchange
        drink += numBottles // numExchange
        numBottles = numBottles - (numBottles // numExchange * numExchange) + empty

    return drink

# =============================================================================
#

def averageWaitingTime(self, customers: List[List[int]]) -> float:
    currentTime = 0
    totalwaitTime = 0

    for arrival, time in customers:
        if currentTime < arrival:
            currentTime = arrival
        waitTime = currentTime + time - arrival
        totalwaitTime += waitTime
        currentTime += time

    return totalwaitTime / len(customers)

# =============================================================================
# 860. Lemonade Change

def lemonadeChange(self, bills: List[int]) -> bool:
    five = 0
    ten = 0
    twenty = 0
    for i in bills:
        if i == 5:
            five += 1
        elif i == 10:
            if five != 0:
                five -= 1
                ten += 1
            else:
                return False
        elif i == 20:
            if five > 0 and ten > 0:
                five -= 1
                ten -= 1
                twenty += 1
            elif five > 2:
                five -= 3
                twenty += 1
            else:
                return False
    return True

# =============================================================================
# 624. Maximum Distance in Arrays

def maxDistance(self, arrays: List[List[int]]) -> int:
    minn = sorted(arrays, key=lambda x: x[0])
    maxx = sorted(arrays, key=lambda x: x[-1], reverse=True)
    if maxx[0] != minn[0]:
        return abs(maxx[0][-1] - minn[0][0])
    return max(abs(maxx[0][-1] - minn[1][0]), abs(maxx[1][-1] - minn[0][0]))

# =============================================================================
# 151. Reverse Words in a String

def reverseWords(self, s: str) -> str:
    x = s.split()
    res = ''
    for i in range(len(x) - 1, -1, -1):
        res += x[i]
        if i != 0:
            res += ' '
    return res

def reverseWords(self, s: str) -> str:
    x = s.split()[::-1]
    return ' '.join(x)

# =============================================================================
# 334. Increasing Triplet Subsequence

def increasingTriplet(self, nums: List[int]) -> bool:
    first = 2 ** 31
    second = 2 ** 31
    for i in nums:
        if i < first:
            first = i
        if first < i < second:
            second = i
        if i > second:
            return True
    return False

# =============================================================================
# 258. Add Digits

def addDigits(self, num: int) -> int:
    res = num
    tmp = res
    while len(str(res)) > 1:
        res = 0
        for i in str(tmp):
            res += int(i)
        tmp = res
    return res

# =============================================================================
# 171. Excel Sheet Column Number

def titleToNumber(self, columnTitle: str) -> int:
    res = 0
    pos = 0
    for i in range(len(columnTitle) - 1, -1, -1):
        res += (26 ** pos) * (ord(columnTitle[i]) - 64)
        pos += 1
    return res

# =============================================================================
# 476. Number Complement

def findComplement(self, num: int) -> int:
    x = format(num, 'b')
    ## format(14, '#b'), format(14, 'b') -->('0b1110', '1110'); f'{14:#b}', f'{14:b}' -->('0b1110', '1110')
    y = ''
    for i in str(x):
        if i == '0':
            y += '1'
        else:
            y += '0'
    res = int(y, base=2)  # из bin в число
    return res

# =============================================================================
# 67. Add Binary

def addBinary(self, a: str, b: str) -> str:
    return bin(
        int(a, 2) + int(b, 2)
    )[2:]

# =============================================================================
# 338. Counting Bits

def countBits(self, n: int) -> List[int]:
    ans = []
    for i in range(n + 1):
        ans.append(bin(i).count('1'))
    return ans

def countBits(self, n: int) -> List[int]:
    res = []
    for i in range(n+1):
        x = 0
        for j in str(format(i, 'b')):
            x += int(j)
        res.append(x)
    return res

# =============================================================================
# 2022. Convert 1D Array Into 2D Array

def construct2DArray(self, original: List[int], m: int, n: int) -> List[List[int]]:
    if n * m != len(original):
        return []
    res = []
    while m > 0:
        tmp = []
        for i in range(n):
            tmp.append(original[0])
            original.pop(0)
        res.append(tmp)
        m = m - 1
    return res

# =============================================================================
# 1945. Sum of Digits of String After Convert

def getLucky(self, s: str, k: int) -> int:
    res = ''
    for i in s:
        res += str(ord(i) - 96) ## или str(ord(x) - ord('a') + 1), ord('a') = 97

    while k > 0:
        tmp = 0
        for i in str(res):
            tmp += int(i)
        res = tmp
        k -= 1
    return res

# =============================================================================
# 88. Merge Sorted Array

# for i in range(n):
#     nums1.pop()
# nums1 += nums2
# nums1.sort()

# =============================================================================
# 3168. Minimum Number of Chairs in a Waiting Room
def minimumChairs(self, s: str) -> int:
    num = 0
    res = 0
    for i in s:
        if i == 'E':
            num += 1
            if num > res:
                res = num
        else:
            num -= 1
    return res

# =============================================================================
# 3099. Harshad Number

def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
    num = ''
    for i in str(x):
        num += i
    summ = 0
    for i in num:
        summ += int(i)

    if x % summ == 0:
        return summ
    return -1

# =============================================================================
# 1668. Maximum Repeating Substring

def maxRepeating(self, sequence: str, word: str) -> int:
    res = 0
    tmp = 0
    x = len(word)
    i = 0
    while i < len(sequence) - x + 1:
        if sequence[i:x + i] == word:
            tmp += 1
            res = max(tmp, res)
            i += x
        elif tmp > 0:
            tmp = 0
            i -= x - 1
        else:
            i += 1
            tmp = 0
    return res

# =============================================================================
# 832. Flipping an Image

def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
    new = []
    for i in image:
        new.append(i[::-1])
    res = []
    for i in new:
        tmp = []
        for j in i:
            if j == 0:
                tmp.append(1)
            else:
                tmp.append(0)
        res.append(tmp)
    return res

# =============================================================================
# 69. Sqrt(x)

def mySqrt(self, x: int) -> int:
    return int(x ** 0.5)

# =============================================================================
# 1984. Minimum Difference Between Highest and Lowest of K Scores

def minimumDifference(self, nums: List[int], k: int) -> int:
    if len(nums) == 1:
        return 0
    nums.sort()
    res = 100000  # float('inf') - бесконечность
    for i in range(len(nums)-k+1):
        diff = nums[i+k-1] - nums[i]
        if diff < res:
            res = diff
            if diff == 0:
                return 0
    return res

# =============================================================================
# 819. Most Common Word

class Solution:
    def count(self, word):
        dic = {}
        for i in word:
            if i not in dic:
                dic[i] = 0
            dic[i] += 1
        return dic

    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        banned = set(banned)
        paragraph = paragraph.lower()
        symbols = "!?',;."
        for i in symbols:
            if i in paragraph:
                paragraph = paragraph.replace(i, ' ')
        paragraph = paragraph.split()

        x = self.count(paragraph)
        new = sorted(list(x.keys()), key=lambda y: -x[y])
        for i in new:
            if i not in banned:
                return i

# =============================================================================
# 3046. Split the Array

def count(self, word):
    dic = {}
    for i in word:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    return dic
def isPossibleToSplit(self, nums: List[int]) -> bool:
    x = self.count(nums)
    for key, val in x.items():
        if val > 2:
            return False
    return True

# =============================================================================
# 1185. Day of the Week

import datetime
class Solution:
    def dayOfTheWeek(self, day: int, month: int, year: int) -> str:
        date = datetime.datetime(year, month, day)
        x = date.weekday()  # определение дня недели по дате
        return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x]

# =============================================================================
# 202. Happy Number

def square(self, num):
    res = 0
    for digit in str(num):
        res += int(digit) ** 2
    return res

def isHappy(self, n: int) -> bool:
    seen = set([n])
    while n != 1:
        n = self.square(n)
        if n in seen:
            return False
        seen.add(n)
    return True

# =============================================================================
# 539. Minimum Time Difference

def findMinDifference(self, timePoints: List[str]) -> int:
    minutes = []
    for i in timePoints:
        x = int(i[0:2]) * 60 + int(i[3:5])
        minutes.append(x)
    minutes.sort()
    minn = 1440
    for i in range(1, len(minutes)):
        minn = min((minutes[i]-minutes[i-1]), minn)
    if len(minutes) > 1:
        x = 1440 - minutes[-1] + minutes[0]
        minn = min(minn, x)
    return minn

# =============================================================================
# 884. Uncommon Words from Two Sentences

def count(self, word):
    dic = {}
    for i in word:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    return dic
def uncommonFromSentences(self, s1: str, s2: str) -> List[str]:
    common = s1 + ' ' + s2
    res = []
    x = self.count(common.split())
    for key, val in x.items():
        if val == 1:
            res.append(key)
    return res

# =============================================================================
# 179. Largest Number

def largestNumber(self, nums: List[int]) -> str:
    new = []
    for i in nums:
        new.append(str(i))

    # new = [str(i) for i in nums]

    new.sort(reverse=True) ## неверная сортировка
    return ''.join(new)

# =============================================================================
# 1556. Thousand Separator

def thousandSeparator(self, n: int) -> str:
    n = str(n)
    res = ''
    step = 0
    for i in range((len(n))-1, -1, -1):
        res = n[i] + res
        step += 1
        if step == 3 and i != 0:
            res = '.' + res
            step = 0
    return res

# =============================================================================
# 290. Word Pattern

def wordPattern(self, pattern: str, s: str) -> bool:
    new = s.split()
    if len(new) != len(pattern):
        return False

    dic = {}
    for i in range(len(new)):
        if new[i] not in dic:
            dic[new[i]] = pattern[i]
        else:
            if dic[new[i]] != pattern[i]:
                return False
    values = list(dic.values())
    return len(values) == len(set(values))

# =============================================================================
# 1805. Number of Different Integers in a String

def numDifferentIntegers(self, word: str) -> int:
    for i in word:
        if not i.isdigit():  ## проверка, что символ не число
            word = word.replace(i, " ")
    word = word.split()
    res = set()
    for i in word:
        res.add(int(i))
    return len(res)

# =============================================================================
# 386. Lexicographical Numbers

def lexicalOrder(self, n: int) -> List[int]:
    return [int(i) for i in sorted([str(i) for i in list(range(1, n+1))])]

def lexicalOrder(self, n: int) -> List[int]:
    new = list(range(1, n+1))
    res = sorted([str(i) for i in new])
    res2 = [int(i) for i in res]  # List comprehension
    return res2

def lexicalOrder(self, n: int) -> List[int]:
    new = list(range(1, n+1))
    res = []
    for i in new:
        res.append(str(i))
    res.sort()
    res2 = []
    for i in res:
        res2.append(int(i))
    return res2

# =============================================================================
# 520. Detect Capital

def detectCapitalUse(self, word: str) -> bool:
    return word.isupper() or word.islower() or word.istitle()  # istitle() - с 1я заглавная, остальные строчные

def detectCapitalUse(self, word: str) -> bool:
    return word.isupper() or word.islower() or (word[0].isupper() and word[1:].islower())

# =============================================================================
# 415. Add Strings

import sys

def addStrings(self, num1: str, num2: str) -> str:
    sys.set_int_max_str_digits(6000)
    return str(int(num1) + int(num2))

# =============================================================================
# 2079. Watering Plants

def wateringPlants(self, plants: List[int], capacity: int) -> int:
    wat = capacity
    i = 0
    steps = 0
    while i < len(plants):
        if wat >= plants[i]:
            wat -= plants[i]
            i += 1
            if i < len(plants) and wat >= plants[i]:
                steps += 1
        else:
            wat = capacity
            steps += (i + 1) * 2 - 1
    steps += 1
    return steps

# =============================================================================
# 1910. Remove All Occurrences of a Substring

def removeOccurrences(self, s: str, part: str) -> str:
    n = len(part)
    while part in s:
        i = s.find(part)
        s = s[:i] + s[i+n:] ## удаление в стринге через срез
    return s

def removeOccurrences(self, s: str, part: str) -> str:
    while part in s:
        s = s.replace(part, "", 1) ## удаление в стринге через замену replace
    return s

# =============================================================================
# 3227. Vowels Game in a String

from collections import Counter
def doesAliceWin(self, s: str) -> bool:
    vowels = 'aeiou'
    counter = Counter(letter for letter in s if letter in vowels)  ## List comprehension с условием. считаем только гласные, вернет словарь
    count_vowels = sum(counter.values())
    return count_vowels != 0

def doesAliceWin(self, s: str) -> bool:
    vowels = 'aeiou'
    for i in s:
        if i in vowels:
            return True
    return False

# =============================================================================
# 2243. Calculate Digit Sum of a String

def digitSum(self, s: str, k: int) -> str:
    new = ''
    while len(s) > k:
        i = 0
        tmp = 0
        for num in s:
            if i < k:
                if tmp is None:
                    tmp = int(num)
                else:
                    tmp += int(num)
                i += 1
                if i == k:
                    new += str(tmp)
                    tmp = None
                    i = 0
        if tmp is not None:
            new += str(tmp)
        s = new
        new = ''
    return s

# =============================================================================
# 1561. Maximum Number of Coins You Can Get

def maxCoins(self, piles: List[int]) -> int:
    res = 0
    x = len(piles)
    piles.sort(reverse=True)
    for i in range(1, x // 3 * 2, 2):
        res += piles[i]
    return res

def maxCoins(self, piles: List[int]) -> int:
    res = 0
    piles.sort(reverse=True)
    while len(piles) > 2:
        piles.pop(0)
        res += piles[0]
        piles.pop(0)
        piles.pop(-1)
    return res

# =============================================================================
# 2433. Find The Original Array of Prefix Xor

def findArray(self, pref: List[int]) -> List[int]:
    res = [pref[0]]
    for i in range(1, len(pref)):
        res.append(pref[i-1] ^ pref[i])  # чтобы расксорить, надо заксорить еще раз. и все :)
    return res

# =============================================================================
# 3158. Find the XOR of Numbers Which Appear Twice

from collections import Counter


def duplicateNumbersXOR(self, nums: List[int]) -> int:
    res = 0
    counter = Counter(nums)
    double = [key for key, value in counter.items() if value > 1]
    if len(double) >= 1:
        res = double[0]
        for i in range(1, len(double)):
            res = double[i] ^ res
    return res

# =============================================================================
# 2390. Removing Stars From a String

def removeStars(self, s: str) -> str:
    stack = []
    for i in s:
        if i != '*':
            stack.append(i)
        else:
            stack.pop()
    return ''.join(stack)


def removeStars(self, s: str) -> str: #медленно
    while '*' in s:
        i = s.index('*')
        s = s[:i - 1] + s[i + 1:]
    return s

# =============================================================================
# 1331. Rank Transform of an Array

def arrayRankTransform(self, arr: List[int]) -> List[int]:
    uniq = set(arr)
    sorted_list = sorted(uniq)  ## метод .sort() изменяет исходник. функция sorted() не изменяет исходник, сохранять в новую переменную
    dic = {}
    for i in range(1, len(sorted_list)+1):
        dic[sorted_list[i-1]] = i
    for i in range(len(arr)):
        arr[i] = dic[arr[i]]
    return arr

# =============================================================================
#1154. Day of the Year

import datetime
class Solution:
    def dayOfYear(self, date: str) -> int:
        year = int(date[:4])
from functools import cache
from typing import List

# =============================================================================
#

# =============================================================================
#

# =============================================================================
#

# =============================================================================
# 2418. Sort the People

# names = ["Mary", "John", "Emma"]
# heights = [180, 165, 170]

def sortPeople(self, names, heights):
    dic = {}
    new_list = []
    for i in range(len(names)):
        dic[heights[i]] = names[i]
    dic = sorted(dic.items(), reverse=True)
    for i in dic:
        new_list.append(i[1])
    return new_list

def sortPeople(self, names: List[str], heights: List[int]) -> List[str]:
    persons = []
    for i in range(len(names)):
        persons.append({"name": names[i], "height": heights[i]})

    persons.sort(key=lambda x: x["height"], reverse=True)
    result = [person["name"] for person in persons]
    return result

# =============================================================================
# 2942. Find Words Containing Character

def findWordsContaining(self, words, x):
    res = []
    for i in range(len(words)):
        if x in words[i]:
            res.append(i)
    return res

# =============================================================================
# 1859. Sorting the Sentence

def sortSentence(self, s):
    s = s.split()
    s.sort(key=xyi)
    result = []
    for i in s:
        i = i[:-1]
        result.append(i)
    result = ' '.join(result)
    return result

def xyi(word):
    return word[-1]

# =============================================================================
# 1684. Count the Number of Consistent Strings

def countConsistentStrings(allowed, words):
    count = 0
    for word in words:
        match = True
        for letter in word:
            if letter not in allowed:
                match = False
        if match:
            count += 1
    print(count)

# =============================================================================
# 1662. Check If Two String Arrays are Equivalent

def arrayStringsAreEqual(self, word1: List[str], word2: List[str]) -> bool:
    return ''.join(word1) == ''.join(word2)

def arrayStringsAreEqual(self, word1, word2):
    new1 = ''
    new2 = ''
    for i in word1:
        new1 += i
    for i in word2:
        new2 += i
    if new1 == new2:
        return True

# =============================================================================
# 344. Reverse String

def reverseString(s):
    new = ''.join(s)
    new = new[::-1]
    res = []
    for i in new:
        res.append(i)
    print(res)

def reverseString1(s):
    s[:] = s[::-1]
    print(s)

def reverseString3(self, s):
    s.reverse()
    print(s)

# =============================================================================
# 2710. Remove Trailing Zeros From a String

def removeTrailingZeros(num):
    return str(int(num[::-1]))[::-1]

# =============================================================================
# 1929. Concatenation of Array

def getConcatenation(self, nums):
    return nums * 2

def getConcatenation(self, nums):
    return nums + nums

def getConcatenation(self, nums):
    nums.extend(nums)
    return nums

# =============================================================================
# 1672. Richest Customer Wealth

def maximumWealth(self, accounts):
    res = []
    for i in accounts:
        res.append(sum(i))
    return max(res)

# =============================================================================
# 2798. Number of Employees Who Met the Target
hours = [0, 1, 2, 3, 4]
target = 2

# Output: 3

def numberOfEmployeesWhoMetTarget(hours, target):
    count = 0
    for i in hours:
        if i >= target:
            count += 1
    return count

# =============================================================================
# 1678. Goal Parser Interpretation
# G -> G
# () -> o
# (al) -> al
command = "G()(al)"

# Output: "Goal"
def interpret(self, command):
    command = command.replace('()', 'o')
    command = command.replace('(al)', 'al')
    return command

def interpret(self, command):
    return command.replace('()', 'o').replace('(al)', 'al')

# =============================================================================
# 1480. Running Sum of 1d Array

# 0)
def runningSum(self, nums):
    res = []
    curr = 0
    for elem in nums:
        curr += elem
        res.append(curr)
    return res

# # 1)
# def sum(self, nums, index):
#     sum = 0
#     for i in range(index + 1):
#         sum = sum + nums[i]
#     return sum

def runningSum(self, nums):
    res = []
    for x in range(len(nums)):  # 10
        elem = nums[x]
        res.append(self.sum(nums, x))
    return res

# 2)

def runningSum(self, nums):
    res = []
    curr = 0
    for x in range(len(nums)):
        elem = nums[x]
        curr += elem
        res.append(curr)
    return res

# 3)
def runningSum(nums):
    res = [nums[0]]
    for x in range(1, len(nums)):
        elem = nums[x]
        new = res[x - 1]
        new += elem
        res.append(new)
    return res

# 4)
def runningSum(self, nums):
    res = [nums[0]]
    for x in range(1, len(nums)):
        elem = nums[x]
        new = res[-1] + elem
        res.append(new)
    return res

# 4)
def runningSum(self, nums):
    res = []
    for x in range(len(nums)):
        elem = nums[x]
        if len(res) > 0:
            prev_sum = res[-1]
        else:
            prev_sum = 0
        res.append(prev_sum + elem)
    return res

# 5) перезапись, так не надо
def runningSum(nums):
    for i in range(1, len(nums)):
        nums[i] = nums[i - 1] + nums[i]
        return nums

# =============================================================================
# 1470. Shuffle the Array
def shuffle(self, nums, n):
    res = []
    for i in range(n):
        el = nums[i]
        sec = nums[i + n]
        res.append(el)
        res.append(sec)
    return res

# 2)

def shuffle(self, nums: List[int], n: int) -> List[int]:
    array1 = nums[:n]
    array2 = nums[n:]
    result = []
    for i in range(n):
        result.append(array1[i])
        result.append(array2[i])
    return result

# =============================================================================
# 1431. Kids With the Greatest Number of Candies
candies = [2, 3, 5, 1, 3]
extraCandies = 3

# Output: [true,true,true,false,true]

def kidsWithCandies(candies, extraCandies):
    res = []
    for i in range(len(candies)):
        el = candies[i] + extraCandies
        if max(candies) <= el:
            greatest = True
        else:
            greatest = False
        res.append(greatest)
    print(res)

def kidsWithCandies(self, candies, extraCandies):
    res = []
    max_el = max(candies)
    for candy in candies:
        el = candy + extraCandies
        if max_el <= el:
            res.append(True)
        else:
            res.append(False)
    return res

# =============================================================================
# 1920. Build Array from Permutation
'''Input: nums = [0,2,1,5,3,4]
Output: [0,1,2,4,5,3]
Explanation: The array ans is built as follows: 
ans = [nums[nums[0]], nums[nums[1]], nums[nums[2]], nums[nums[3]], nums[nums[4]], nums[nums[5]]]
    = [nums[0], nums[2], nums[1], nums[5], nums[3], nums[4]]
    = [0,1,2,4,5,3]'''

def buildArray(self, nums):
    res = []
    for i in range(len(nums)):
        el = nums[i]
        x = nums[el]
        res.append(x)
    return res

# =============================================================================
# 1281. Subtract the Product and Sum of Digits of an Integer

n = 564

def subtractProductAndSum(self, n):
    new = str(n)
    summ = 0
    product = 1
    for i in range(len(new)):
        el = int(new[i])
        summ += el
        product *= el
    return product - summ

def subtractProductAndSum(self, n):
    new = str(n)
    summ = 0
    product = 1
    for el in new:
        el = int(el)
        summ += el
        product *= el
    return product - summ

# =============================================================================
# 1512. Number of Good Pairs

def numIdenticalPairs(self, nums):
    res = 0
    for i in range(len(nums)):
        for j in range(len(nums)):
            if nums[i] == nums[j] and i < j:
                res += 1
    return res

# =============================================================================
# 1365. How Many Numbers Are Smaller Than the Current Number

def smallerNumbersThanCurrent(self, nums):
    res = []
    for i in range(len(nums)):
        count = 0
        for j in range(len(nums)):
            if j != i and nums[j] < nums[i]:
                count += 1
        res.append(count)
    return res

def smallerNumbersThanCurrent(self, nums):
    new = sorted(nums)
    res = []
    for i in nums:
        res.append(new.index(i))
    return res

# =============================================================================
# 1603. Design Parking System

# Input
# ["ParkingSystem", "addCar", "addCar", "addCar", "addCar"]
# [[1, 1, 0], [1], [2], [3], [1]]
# Output
# [null, true, true, false, false]

class ParkingSystem(object):
    def __init__(self, big, medium, small):
        self.big = big
        self.medium = medium
        self.small = small

    def addCar(self, carType):
        if carType == 1:
            if self.big > 0:
                self.big -= 1
                return True
            else:
                return False
        elif carType == 2:
            if self.medium > 0:
                self.medium -= 1
                return True
            else:
                return False
        else:
            if self.small > 0:
                self.small -= 1
                return True
            else:
                return False

# 2)

class ParkingSystem(object):
    def __init__(self, big, medium, small):
        self.available_slots = {
            1: big,
            2: medium,
            3: small
        }

    def addCar(self, carType):
        if self.available_slots[carType] > 0:
            self.available_slots[carType] -= 1
            return True
        return False

# Your ParkingSystem object will be instantiated and called as such:
# obj = ParkingSystem(big, medium, small)
# param_1 = obj.addCar(carType)

# =============================================================================
# 2769. Find the Maximum Achievable Number

def theMaximumAchievableX(self, num, t):
    return num + t * 2

# =============================================================================
# 2824. Count Pairs Whose Sum is Less than Target

def countPairs(self, nums, target):
    count = 0
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] < target:
                count += 1
    return count

def countPairs(self, nums, target):
    count = 0
    n = len(nums)
    for i in range(n):
        for j in range(n):
            if 0 <= i < j < n and nums[i] + nums[j] < target:
                count += 1
    return count

# =============================================================================
# 2160. Minimum Sum of Four Digit Number After Splitting Digits
num = 2932

# Output: 52

def minimumSum(self, num):
    new = sorted(str(num))
    return int(new[0] + new[2]) + int(new[1] + new[3])

def minimumSum(self, num):
    new = sorted(str(num))
    new1 = []
    new2 = []
    new1.extend([new[0], new[2]])
    new2.extend([new[1], new[3]])
    new1 = int(''.join(new1))
    new2 = int(''.join(new2))
    res = new1 + new2
    return res

# =============================================================================
# 1313. Decompress Run-Length Encoded List

def decompressRLElist(self, nums):
    res = []
    for index in range(0, len(nums), 2):
        freq = nums[index]
        val = nums[index + 1]
        res.extend([val] * freq)
    return res

# =============================================================================
# 2859. Sum of Values at Indices With K Set Bits

def sumIndicesWithKSetBits(self, nums, k):
    res = 0
    for i in range(len(nums)):
        i_bin = bin(i)
        count_1 = 0
        for x in i_bin:
            if x == '1':
                count_1 += 1
        if count_1 == k:
            res += nums[i]
    return res

# 2)
def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
    ans = 0
    n = len(nums)
    for i in range(n):
        if bin(i)[2:].count("1") == k:  ## checking set bits in binary num
            ans += nums[i]
    return (ans)

# =============================================================================
# 1720. Decode XORed Array

def decode(self, encoded, first):
    res = [first]
    for i in range(len(encoded)):
        x = res[i] ^ encoded[i]
        res.append(x)
    return res

# =============================================================================
# 1389. Create Target Array in the Given Order

def createTargetArray(self, nums, index):
    res = []
    for i in range(len(nums)):
        res.insert(index[i], nums[i])
    return res

def createTargetArray(self, nums, index):
    arr = []
    for n, i in zip(nums, index):
        arr.insert(i, n)
    return arr

# =============================================================================
# 1486. XOR Operation in an Array
n = 4
start = 3

# Output: 8

def xorOperation(n, start):
    res = start
    for i in range(1, n):
        res = res ^ (start + 2 * i)
    return res

# =============================================================================
# 1342. Number of Steps to Reduce a Number to Zero

def numberOfSteps(self, num):
    steps = 0
    while num != 0:
        if num % 2 == 0:
            num /= 2
            steps += 1
        else:
            num -= 1
            steps += 1
    return steps

# =============================================================================
# 2652. Sum Multiples

def sumOfMultiples(self, n):
    res = 0
    for i in range(3, n + 1):
        if i % 3 == 0 or i % 5 == 0 or i % 7 == 0:
            res += i
    return res

# =============================================================================
# 2520. Count the Digits That Divide a Number

def countDigits(self, num):
    res = 0
    for i in str(num):
        if num % int(i) == 0:
            res += 1
    return res

# =============================================================================
# 2535. Difference Between Element Sum and Digit Sum of an Array

def differenceOfSum(self, nums):
    sum1 = sum(nums)
    sum2 = 0
    for i in nums:
        for j in str(i):
            sum2 += int(j)
    return abs(sum1 - sum2)

# =============================================================================
# 1656. Design an Ordered Stream

class OrderedStream(object):
    def __init__(self, n):
        self.n = n
        self.new = [None] * n
        self.position = 0

    def insert(self, idKey, value):
        self.new[idKey - 1] = value
        res = []
        while self.position < self.n and self.new[self.position] is not None:
            res.append(self.new[self.position])
            self.position += 1
        return res

class OrderedStream:
    def __init__(self, n: int):
        self.pairs = dict()
        self.position = 0

    def insert(self, idKey: int, value: str) -> list[str]:
        self.pairs[idKey] = value

        res = list()
        while self.position + 1 in self.pairs:
            self.position += 1

            res.append(self.pairs[self.position])

        return res

# =============================================================================
# 2325. Decode the Message

class Solution(object):
    def decodeMessage(self, key, message):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        uniq_key = ''
        res = ''
        for i in key:
            if i not in uniq_key and i != ' ':
                uniq_key += i
        for i in message:
            if i == ' ':
                res += ' '
            else:
                x = uniq_key.index(i)
                res += alphabet[x]
        return res

class Solution(object):
    def decodeMessage(self, key, message):
        alphabet = ' abcdefghijklmnopqrstuvwxyz'  ## пробел заменяем на пробел
        uniq_key = ' '
        res = ''
        for i in key:
            if i not in uniq_key:
                uniq_key += i
        for i in message:
            x = uniq_key.index(i)
            res += alphabet[x]
        return res

class Solution:
    def decodeMessage(self, key: str, message: str) -> str:
        mapping = {' ': ' '}
        i = 0
        res = ''
        letters = 'abcdefghijklmnopqrstuvwxyz'

        for char in key:
            if char not in mapping:
                mapping[char] = letters[i]
                i += 1

        for char in message:
            res += mapping[char]

        return res

# =============================================================================
# 1913. Maximum Product Difference Between Two Pairs

def maxProductDifference(self, nums):
    new = sorted(nums)
    num1 = new[0] * new[1]
    num2 = new[-1] * new[-2]
    return num2 - num1

# =============================================================================
# 557. Reverse Words in a String III

def reverseWords(self, s):
    res = []
    new = s.split(' ')
    for i in new:
        i = i[::-1]
        res.append(i)
    res = ' '.join(res)
    return res

def reverseWords(self, s: str) -> str:
    s = s.split(' ')
    new = ''
    for word in s:
        new += word[::-1] + ' '

    return new[:-1]

# =============================================================================
# 2828. Check if a String Is an Acronym of Words

def isAcronym(self, words, s):
    res = False
    new = ''
    for i in words:
        new += i[0]
    if s == new:
        res = True
    return res

def isAcronym(self, words, s):
    new = ''
    for i in words:
        new += i[0]
    return s == new

# =============================================================================
# 804. Unique Morse Code Words

def uniqueMorseRepresentations(self, words):
    alph = 'abcdefghijklmnopqrstuvwxyz'
    morse = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--", "-.", "---",
             ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."]
    dic = {}
    for i in range(len(alph)):
        dic[alph[i]] = morse[i]
    new = []
    for word in words:
        new1 = ''
        for let in word:
            new1 += dic[let]
        new.append(new1)
    return len(set(new))

def uniqueMorseRepresentations(self, words):
    alph = 'abcdefghijklmnopqrstuvwxyz'
    morse = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--", "-.", "---",
             ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."]
    morse_dict = dict(zip(alph, morse))
    new = []
    for word in words:
        new1 = ''
        for let in word:
            new1 += morse_dict[let]
        new.append(new1)
    return len(set(new))

# =============================================================================
# 1732. Find the Highest Altitude

def largestAltitude(self, gain):
    res = 0
    new = 0
    for i in gain:
        new += i
        if new > res:
            res = new
    return res

def largestAltitude(self, gain: List[int]) -> int:
    highest_point = 0
    prev_altitude = 0
    for i in gain:
        prev_altitude += i
        highest_point = max(highest_point, prev_altitude)
    return highest_point

# =============================================================================
# 1464. Maximum Product of Two Elements in an Array

def maxProduct(self, nums):
    new = sorted(nums, reverse=True)
    res = (new[0] - 1) * (new[1] - 1)
    return res

def maxProduct(self, nums):
    new = sorted(nums)
    return (new[-1] - 1) * (new[-2] - 1)

# =============================================================================
# 1323. Maximum 69 Number

def maximum69Number(self, num):
    new = str(num)
    res = ''
    change = False
    for i in range(len(new)):
        if new[i] == '6':
            if change:
                res += '6'
            else:
                res += '9'
                change = True
        else:
            res += '9'
    return int(res)

def maximum69Number(self, num: int) -> int:
    return int(str(num).replace('6', '9', 1))

def maximum69Number(self, num):
    temp = num
    s = (str(num))
    for i in range(len(s)):
        if s[i] == "6":
            val = (int(s[:i] + "9" + s[i + 1:]))
        else:
            val = (int(s[:i] + "6" + s[i + 1:]))
        temp = max(temp, val)
    return temp

def maximum69Number(self, num):
    new = list(str(num))
    if '6' in new:
        idx = new.index('6')
        new[idx] = '9'
    return int(''.join(new))

# =============================================================================
# 2427. Number of Common Factors

def commonFactors(self, a, b):
    res = 0
    new = min(a, b)
    for i in range(1, new + 1):
        if a % i == 0 and b % i == 0:
            res += 1
    return res

# =============================================================================
# 728. Self Dividing Numbers

def selfDividingNumbers(self, left, right):
    res = []
    for i in range(left, right + 1):
        count = 0
        for j in str(i):
            if int(j) != 0 and int(i) % int(j) == 0:
                count += 1
        if count == len(str(i)):
            res.append(i)
    return res

class Solution(object):
    def isDividingNumber(self, number):
        for digit in str(number):
            if int(digit) == 0 or number % int(digit) != 0:
                return False
        return True

    def selfDividingNumbers(self, left, right):
        res = []
        for number in range(left, right + 1):
            if self.isDividingNumber(number):
                res.append(number)
        return res

# =============================================================================
# 1716. Calculate Money in Leetcode Bank

def totalMoney(self, n):
    res = 0
    save = 0
    start = 0
    for i in range(n):
        if i % 7 == 0:
            start += 1
            save = start
        else:
            save += 1
        res += save
    return res

# =============================================================================
# 2367. Number of Arithmetic Triplets

def arithmeticTriplets(self, nums, diff):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            for k in range(j + 1, len(nums)):
                if nums[j] - nums[i] == diff and nums[k] - nums[j] == diff:
                    count += 1
    return count

# =============================================================================
# 2006. Count Number of Pairs With Absolute Difference K

def countKDifference(self, nums, k):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] - nums[j] == k or nums[i] - nums[j] == k * -1:
                count += 1
    return count

def countKDifference(self, nums, k):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if abs(nums[i] - nums[j]) == k:  # abs() - модуль
                count += 1
    return count

def countKDifference(self, nums, k):
    count = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] - nums[j] == k or nums[j] - nums[i] == k:
                count += 1
    return count

# =============================================================================
# 2574. Left and Right Sum Differences

def leftRightDifference(self, nums):
    res = []
    sum_left = 0
    sum_rigth = sum(nums)
    for i in nums:
        sum_rigth -= i
        res.append(abs(sum_left - sum_rigth))
        sum_left += i
    return res

# так писать нельзя
def leftRightDifference(self, nums):
    res = []
    for i in range(len(nums)):
        left = self.calcSum(nums, 0, i - 1)
        right = self.calcSum(nums, i + 1, len(nums))
        res.append(abs(right - left))
    return res

def calcSum(self, nums, left, right):
    return sum(nums[max(left, 0):min(len(nums), right) + 1])

# =============================================================================
# 2974. Minimum Number Game

def numberGame(self, nums):
    new = sorted(nums)
    res = []
    for i in range(0, len(nums), 2):
        res.append(new[i + 1])
        res.append(new[i])
    return res

# =============================================================================
# 1688. Count of Matches in Tournament

def numberOfMatches(self, n):
    total = 0
    teams = n
    while teams > 1:
        if teams % 2 != 0:
            total += (teams - 1) / 2
            teams = (teams - 1) / 2 + 1
        else:
            teams = teams / 2
            total += teams
    return total

def numberOfMatches(self, n):
    return n - 1

def numberOfMatches(self, n: int) -> int:
    ans = 0
    while n > 1:
        ans += (n // 2)
        n = (n // 2) + (n % 2)
    return ans

def numberOfMatches(self, n):  # рекурсия
    if n == 1:
        return 0

    if n % 2 == 0:
        numberOfMatchesInCurrentRound = n / 2
        numberOfTeamsForNextRound = n / 2
    else:
        numberOfMatchesInCurrentRound = (n - 1) / 2
        numberOfTeamsForNextRound = (n - 1) / 2 + 1

    return numberOfMatchesInCurrentRound + self.numberOfMatches(numberOfTeamsForNextRound)

def numberOfMatches(self, n):  # рекурсия
    if n == 1:
        return 0

    if n % 2 == 0:
        return n / 2 + self.numberOfMatches(n / 2)
    else:
        return (n - 1) / 2 + self.numberOfMatches((n - 1) / 2 + 1)

# =============================================================================
# 1588. Sum of All Odd Length Subarrays

def sumOddLengthSubarrays(self, arr: List[int]) -> int:
    summ = 0
    n = len(arr)
    for srez in range(1, n + 1, 2):
        for index in range(n - srez + 1):
            subarray = arr[index:index + srez]
            summ += sum(subarray)
    return summ

def sumOddLengthSubarrays(self, arr: List[int]) -> int:
    s = 0
    for i in range(len(arr)):
        for j in range(i, len(arr), 2):
            s += sum(arr[i:j + 1])
    return s

# =============================================================================
# 9. Palindrome Number

def isPalindrome(x):
    def isPalindrome(self, x):
        return str(x) == str(x)[::-1]

def isPalindrome(self, x):
    if x == 0:
        return True
    if x < 0 or x % 10 == 0:
        return False

    half = 0
    while half < x:
        half = (x % 10) + half * 10
        x = x // 10
    if half > x:
        half //= 10
    return half == x

def isPalindrome(self, x):
    if x < 0:
        return False
    rev = 0
    new_x = x
    while new_x > 0:
        rev = (new_x % 10) + rev * 10
        new_x = new_x // 10
    return rev == x

# =============================================================================
# 2810. Faulty Keyboard

def finalString(self, s):
    res = ''
    for i in range(len(s)):
        if s[i] == 'i':
            res = res[::-1]
        else:
            res += s[i]
    return res

# =============================================================================
# 2956. Find Common Elements Between Two Arrays

def findIntersectionValues(self, nums1, nums2):
    count1 = 0
    count2 = 0
    for i in range(len(nums1)):
        if nums1[i] in nums2:
            count1 += 1
    for i in range(len(nums2)):
        if nums2[i] in nums1:
            count2 += 1
    res = [count1, count2]
    return res

# =============================================================================
# 2656. Maximum Sum With Exactly K Elements

def maximizeSum(self, nums, k):
    maxx = max(nums)
    res = maxx
    for i in range(k - 1):
        maxx += 1
        res += maxx
    return res

def maximizeSum(self, nums: List[int], k: int) -> int:
    return k * max(nums) + k * (k - 1) // 2

# =============================================================================
# 1534. Count Good Triplets

def countGoodTriplets(self, arr, a, b, c):
    count = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if abs(arr[i] - arr[j]) <= a and abs(arr[j] - arr[k]) <= b and abs(arr[i] - arr[k]) <= c:
                    count += 1
    return count

def countGoodTriplets(self, arr, a, b, c):
    count = 0
    n = len(arr)
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            if abs(arr[i] - arr[j]) <= a:
                for k in range(j + 1, n):
                    if abs(arr[j] - arr[k]) <= b and abs(arr[i] - arr[k]) <= c:
                        count += 1
    return count

# =============================================================================
# 1844. Replace All Digits with Characters

def replaceDigits(self, s):
    def shift(letter, num):
        x = ord(
            letter)  ## ord()  ordinal(порядковый номер) возвращает юникодовый код символа. между большой и маленькой англ! буквой 32 символа
        return chr(x + num)

    res = ''
    for i in range(1, len(s), 2):
        y = shift(s[i - 1], int(s[i]))
        res += s[i - 1]
        res += y
    if len(s) % 2 != 0:
        res += s[-1]
    return res

# =============================================================================
# 1572. Matrix Diagonal Sum

def diagonalSum(self, mat):
    res = 0
    for i in range(len(mat)):
        res += mat[i][i]
        res += mat[i][-i - 1]
    if len(mat) % 2 != 0:
        x = len(mat) // 2
        res -= mat[x][x]
    return res

# =============================================================================
# 2315. Count Asterisks
# 1)
def countAsterisks(self, s):
    count = 0
    in_bars = False
    for i in range(len(s)):
        if s[i] == '|':
            in_bars = not in_bars
        if not in_bars:
            if s[i] == '*':
                count += 1
    return count

# 2)
def countAsterisks(self, s):
    count = 0
    new = s.split('|')
    for i in range(0, len(new), 2):
        for j in new[i]:
            if j == '*':
                count += 1
    return count

# =============================================================================
# 2913. Subarrays Distinct Element Sum of Squares I

def sumCounts(self, nums):
    res = 0
    for len_sub in range(1, len(nums) + 1):
        for index in range(len(nums) - len_sub + 1):
            sub = nums[index:index + len_sub]
            res += len(set(sub)) ** 2
    return res

# =============================================================================
# 1863. Sum of All Subset XOR Totals

from itertools import combinations

class Solution(object):
    def subsetXORSum(self, nums):
        res = 0
        for i in range(1, len(nums) + 1):  # длина
            for sub in combinations(nums, i):
                first_sub = 0
                for num in sub:
                    first_sub = first_sub ^ num
                res += first_sub
        return res

# =============================================================================
# 1436. Destination City

def destCity(self, paths):
    new1 = []
    new2 = []
    for i in paths:
        new1.append(i[0])
        new2.append(i[1])
    for i in new2:
        if i not in new1:
            return i

# =============================================================================
# 2485. Find the Pivot Integer

def pivotInteger(self, n):  # очень медленно
    if n == 1:
        return 1
    for i in range(2, n + 1):
        first = sum(list(range(1, i + 1)))
        sec = sum(list(range(i, n + 1)))
        if first == sec:
            return i
    return -1

def pivotInteger(self, n):  # в обратную сторону, тоже медленно
    if n == 1:
        return 1
    for i in range(n, 1, -1):
        first = sum(range(1, i + 1))
        sec = sum(range(i, n + 1))
        if first == sec:
            return i
    return -1

## через формулу арифметической прогрессии
def pivotInteger(self, n):
    temp = (n * n + n) // 2
    sq = int(math.sqrt(temp))
    if sq * sq == temp:
        return sq
    return -1

# =============================================================================
# 2000. Reverse Prefix of Word

def reversePrefix(self, word, ch):
    for i in range(len(word)):
        if word[i] == ch:
            new = word[:i + 1][::-1] + word[i + 1:]
            return new
    return word

# =============================================================================
# 1967. Number of Strings That Appear as Substrings in Word

def numOfStrings(self, patterns, word):
    count = 0
    for i in range(len(patterns)):
        if patterns[i] in word:
            count += 1
    return count

# =============================================================================
# 1351. Count Negative Numbers in a Sorted Matrix

def countNegatives(self, grid):
    count = 0
    for row in grid:
        for val in row:
            if val < 0:
                count += 1
    return count

def countNegatives(self, grid):
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] < 0:
                count += 1
    return count

# =============================================================================
# 1282. Group the People Given the Group Size They Belong To

def groupThePeople(self, groupSizes):
    dic = {}
    for i in range(len(groupSizes)):
        x = groupSizes[i]
        if x not in dic:
            dic[x] = []
        dic[x].append(i)
    res = []
    for key, value in dic.items():
        tmp = []
        for elem in value:
            tmp.append(elem)
            if len(tmp) == key:
                res.append(tmp)
                tmp = []
    return res

# =============================================================================
# 2778. Sum of Squares of Special Elements

def sumOfSquares(self, nums):
    res = 0
    n = len(nums)
    for i in range(1, n + 1):
        if n % i == 0:
            res += nums[i - 1] ** 2
    return res

def sumOfSquares(self, nums):
    res = 0
    n = len(nums)
    for i in range(n):
        if n % (i + 1) == 0:
            res += nums[i] ** 2
    return res

# =============================================================================
# 1309. Decrypt String from Alphabet to Integer Mapping

def freqAlphabets(self, s):
    alp = {'1': 'a',
           '2': 'b',
           '3': 'c',
           '4': 'd',
           '5': 'e',
           '6': 'f',
           '7': 'g',
           '8': 'h',
           '9': 'i',
           '10#': 'j',
           '11#': 'k',
           '12#': 'l',
           '13#': 'm',
           '14#': 'n',
           '15#': 'o',
           '16#': 'p',
           '17#': 'q',
           '18#': 'r',
           '19#': 's',
           '20#': 't',
           '21#': 'u',
           '22#': 'v',
           '23#': 'w',
           '24#': 'x',
           '25#': 'y',
           '26#': 'z'}
    res = ''
    i = 0
    while i < len(s):
        if i + 2 < len(s) and s[i + 2] == '#':
            res += alp[s[i:i + 3]]
            i += 3
        else:
            res += alp[s[i]]
            i += 1
    return res

# =============================================================================
# 2176. Count Equal and Divisible Pairs in an Array

def countPairs(self, nums, k):
    res = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j] and (i * j) % k == 0:
                res += 1
    return res

# =============================================================================
# 1768. Merge Strings Alternately

def mergeAlternately(self, word1, word2):
    res = ''
    maxx = max(len(word1), len(word2))
    for i in range(maxx):
        if i < len(word1):
            res += word1[i]
        if i < len(word2):
            res += word2[i]
    return res

def mergeAlternately(self, word1, word2):
    res = ''
    minn = min(len(word1), len(word2))
    for i in range(minn):
        res += word1[i]
        res += word2[i]
    if len(word1) > len(word2):
        res += word1[len(word2):]
    if len(word1) < len(word2):
        res += word2[len(word1):]
    return res

from itertools import zip_longest

def mergeAlternately(self, word1: str, word2: str) -> str:
    res = ''
    for i, j in zip_longest(word1, word2, fillvalue=''):
        res += i
        res += j
    return res

def mergeAlternately(self, word1: str, word2: str) -> str:
    res = ''
    maxx = max(len(word1), len(word2))
    for i in range(maxx):
        try:
            res += word1[i]
        except IndexError:
            pass
        try:
            res += word2[i]
        except IndexError:
            pass
    return res

# =============================================================================
# 3019. Number of Changing Keys

def countKeyChanges(self, s):
    s = s.lower()
    res = 0
    for i in range(1, len(s)):
        if s[i] != s[i - 1]:
            res += 1
    return res

# =============================================================================
# 1748. Sum of Unique Elements

def sumOfUnique(self, nums):
    uniq = []
    for i in range(len(nums)):
        if nums[i] not in nums[0:i] and nums[i] not in nums[i + 1:]:
            uniq.append(nums[i])
    return sum(uniq)

def sumOfUnique(self, nums):
    uniq = []
    for i in nums:
        if nums.count(i) == 1:
            uniq.append(i)
    return sum(uniq)

# =============================================================================
# 1475. Final Prices With a Special Discount in a Shop

def finalPrices(self, prices):
    for i in range(len(prices) - 1):
        for j in range(i + 1, len(prices)):
            if prices[i] >= prices[j]:
                prices[i] = prices[i] - prices[j]
                break
    return prices

prices = [10, 1, 1, 6]

def finalPrices(self, prices):
    for i in range(len(prices) - 1):
        j = i + 1
        while j < len(prices) and prices[i] < prices[j]:
            j += 1
        if j < len(prices) and prices[i] >= prices[j]:
            prices[i] = prices[i] - prices[j]
    return prices

# =============================================================================
# 1979. Find Greatest Common Divisor of Array

def findGCD(self, nums):
    maxx = max(nums)
    minn = min(nums)
    i = minn
    while i > 0:
        if maxx % i == 0 and minn % i == 0:
            return i
        else:
            i -= 1

def findGCD(self, nums):
    maxx = max(nums)
    minn = min(nums)
    for i in range(minn, 0, -1):  ## убывающий цикл, на уменьшение. цикл в обратную сторону
        if maxx % i == 0 and minn % i == 0:
            return i

# =============================================================================
# 1374. Generate a String With Characters That Have Odd Counts

def generateTheString(self, n):
    if n % 2 == 0:
        res = 'a' * (n - 1) + 'b'
    else:
        res = 'a' * n
    return res

# =============================================================================
# 2185. Counting Words With a Given Prefix

def prefixCount(self, words, pref):
    count = 0
    for word in words:
        if word[:len(pref)] == pref:
            count += 1
    return count

def prefixCount(self, words, pref):
    count = 0
    for word in words:
        if word.startswith(pref):  ## метод проверяет с чего начинается
            count += 1
    return count

# =============================================================================
# 2500. Delete Greatest Value in Each Row

def deleteGreatestValue(self, grid):
    count = 0
    while len(grid[0]) > 0:
        tmp = 0
        for i in range(len(grid)):
            if tmp < max(grid[i]):
                tmp = max(grid[i])
            grid[i].remove(max(grid[i]))  ## удалить первое найденное значение, pop() - удаление по индексу
        count += tmp
    return count

# =============================================================================
# 807. Max Increase to Keep City Skyline

def maxIncreaseKeepingSkyline(self, grid):
    res = 0
    n = len(grid)
    maxx = [0] * n

    for i in range(n):
        for j in range(n):
            if maxx[j] < grid[i][j]:
                maxx[j] = grid[i][j]

    for i in range(n):
        for j in range(n):
            minn = min(maxx[j], max(grid[i]))
            if grid[i][j] < minn:
                res = res + (minn - grid[i][j])
    return res

def maxIncreaseKeepingSkyline(self, grid):
    res = 0
    n = len(grid)
    max_col = [0] * n
    max_row = [0] * n

    for i in range(n):
        for j in range(n):
            max_col[j] = max(max_col[j], grid[i][j])
            max_row[i] = max(max_row[i], grid[i][j])

    for i in range(n):
        for j in range(n):
            minn = min(max_col[j], max_row[i])
            if grid[i][j] < minn:
                res = res + (minn - grid[i][j])
    return res

# =============================================================================
# 2215. Find the Difference of Two Arrays

def findDifference(self, nums1, nums2):
    res = [[], []]
    x = set(nums1)
    y = set(nums2)
    for i in x:
        if i not in y:
            res[0].append(i)
    for i in y:
        if i not in x:
            res[1].append(i)
    return res

# =============================================================================
# 1725. Number Of Rectangles That Can Form The Largest Square

def countGoodRectangles(self, rectangles):
    new = []
    for i in rectangles:
        x = min(i)
        new.append(x)
    y = max(new)
    res = new.count(y)
    return res

def countGoodRectangles(self, rectangles):
    new = []
    for i in rectangles:
        new.append(min(i))
    return new.count(max(new))

# =============================================================================
# 1295. Find Numbers with Even Number of Digits

def findNumbers(self, nums):
    count = 0
    for i in nums:
        if len(str(i)) % 2 == 0:
            count += 1
    return count

# =============================================================================
# 2032. Two Out of Three

def twoOutOfThree(self, nums1, nums2, nums3):
    res = []
    uniq = set(nums1 + nums2 + nums3)
    for i in uniq:
        tmp = 0
        if i in nums1:
            tmp += 1
        if i in nums2:
            tmp += 1
        if tmp > 1:
            res.append(i)
            continue
        if i in nums3:
            tmp += 1
        if tmp > 1:
            res.append(i)
    return res

# =============================================================================
# 2089. Find Target Indices After Sorting Array

def targetIndices(self, nums, target):
    res = []
    x = sorted(nums)
    for i in range(len(nums)):
        if x[i] == target:
            res.append(i)
    return res

# =============================================================================
# 2864. Maximum Odd Binary Number

def maximumOddBinaryNumber(self, s):
    x = ''.join(sorted(s, reverse=True))
    return x[1:] + x[0]

# =============================================================================
# 977. Squares of a Sorted Array

def sortedSquares(self, nums):
    res = []
    for i in nums:
        res.append(i ** 2)
    return sorted(res)

def sortedSquares(self, nums):  ## за один проход, но insert медленный
    res = []
    left = 0
    right = len(nums) - 1
    while left <= right:
        if nums[left] ** 2 > nums[right] ** 2:
            res.insert(0, nums[left] ** 2)
            left += 1
        else:
            res.insert(0, nums[right] ** 2)
            right -= 1
    return res

def sortedSquares(self, nums):
    res = []
    left = 0
    right = len(nums) - 1
    while left <= right:
        if nums[left] ** 2 > nums[right] ** 2:
            res.append(nums[left] ** 2)
            left += 1
        else:
            res.append(nums[right] ** 2)
            right -= 1
    res.reverse()
    return res

def sortedSquares(self, nums):
    res = [0] * len(nums)
    left = 0
    right = len(nums) - 1
    for position in range(len(nums) - 1, -1, -1):
        lft_sqr = nums[left] ** 2
        rght_sqr = nums[right] ** 2
        if lft_sqr > rght_sqr:
            res[position] = lft_sqr
            left += 1
        else:
            res[position] = rght_sqr
            right -= 1
    return res

# =============================================================================
# 1941. Check if All Characters Have Equal Number of Occurrences

def areOccurrencesEqual(self, s: str) -> bool:
    uniq = list(s)
    uniq = set(uniq)
    count = s.count(s[0])
    for i in uniq:
        if s.count(i) != count:
            return False
    return True

def areOccurrencesEqual(self, s: str) -> bool:
    dic = {}
    for i in s:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    return len(set(dic.values())) == 1

def areOccurrencesEqual(self, s: str) -> bool:
    count = []
    for i in set(s):
        count.append(s.count(i))
    return len(set(count)) == 1

# =============================================================================
# 1207. Unique Number of Occurrences

def uniqueOccurrences(self, arr: List[int]) -> bool:
    dic = {}
    for i in arr:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    return len(dic.values()) == len(set(dic.values()))

# =============================================================================
# 1750. Minimum Length of String After Deleting Similar Ends

def minimumLength(s: str) -> int:
    while len(s) > 1 and s[0] == s[-1]:
        sim = s[0]
        while len(s) > 0 and s[0] == sim:
            s = s[1:]
        while len(s) > 0 and s[-1] == sim:
            s = s[:-1]
    return len(s)

# =============================================================================
# 2733. Neither Minimum nor Maximum

def findNonMinOrMax(self, nums: List[int]) -> int:
    if len(nums) >= 2:
        nums.remove(max(nums))
        nums.remove(min(nums))
        if len(nums) > 0:
            return nums[0]
    return -1

def findNonMinOrMax(self, nums: List[int]) -> int:
    if len(nums) > 2:
        nums = sorted(nums)
        return nums[1]
        return -1

# =============================================================================
# 2951. Find the Peaks

def findPeaks(self, mountain: List[int]) -> List[int]:
    res = []
    for i in range(1, len(mountain) - 1):
        if mountain[i] > mountain[i - 1] and mountain[i] > mountain[i + 1]:
            res.append(i)
    return res

# =============================================================================
# 3005. Count Elements With Maximum Frequency

def maxFrequencyElements(self, nums: List[int]) -> int:
    dic = {}
    for i in nums:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    maxx = max(dic.values())
    count = 0
    for value in dic.values():
        if value == maxx:
            count += maxx
    return count

# =============================================================================
# 2540. Minimum Common Value

def getCommon(self, nums1: List[int], nums2: List[int]) -> int:
    y = set(nums2)
    for i in nums1:
        if i in y:
            return i
    return -1

# =============================================================================
# 349. Intersection of Two Arrays

def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
    res = []
    nums2 = set(nums2)
    for i in nums1:
        if i in nums2:
            res.append(i)
    return set(res)

# =============================================================================
# 791. Custom Sort String

def customSortString(order: str, s: str) -> str:
    res = ''
    for i in range(len(order)):
        while order[i] in s:
            res += order[i]
            s = s.replace(order[i], '', 1)
    return res + s

# =============================================================================
# 905. Sort Array By Parity

def sortArrayByParity(self, nums: List[int]) -> List[int]:
    res = []
    for i in nums:
        if i % 2 != 0:
            res.append(i)
        else:
            res.insert(0, i)
    return res

def sortArrayByParity(self, nums: List[int]) -> List[int]:
    even = []
    odd = []
    for i in nums:
        if i % 2 == 0:
            even.append(i)
        else:
            odd.append(i)
    return even + odd

# =============================================================================
# 1304. Find N Unique Integers Sum up to Zero

def sumZero(self, n: int) -> List[int]:
    res = []
    for i in range(1, n // 2 + 1):
        res.append(i)
        res.append(-i)
    if n % 2 != 0:
        res.append(0)
    return res

# =============================================================================
# 961. N-Repeated Element in Size 2N Array

def repeatedNTimes(self, nums: List[int]) -> int:
    dic = {}
    n = len(nums) // 2
    for i in nums:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    for key, val in dic.items():
        if val == n:
            return int(key)

# =============================================================================
# 2678. Number of Senior Citizens

def countSeniors(self, details: List[str]) -> int:
    count = 0
    for i in details:
        if int(i[-4:-2]) > 60:
            count += 1
    return count

# =============================================================================
# 525. Contiguous Array

def findMaxLength(nums: List[int]) -> int:
    class Solution:
        def findMaxLength(self, nums: List[int]) -> int:
            dic = {0: -1}
            summ = 0
            i = 0
            max_len = 0
            while i < len(nums):
                if nums[i] == 1:
                    summ += 1
                if nums[i] == 0:
                    summ -= 1
                if summ not in dic:
                    dic[summ] = i
                else:
                    starti = dic[summ]
                    lenght = i - starti
                    if lenght > max_len:
                        max_len = lenght
                i += 1
            return max_len

# =============================================================================
# 1051. Height Checker

def heightChecker(self, heights: List[int]) -> int:
    sort = sorted(heights)
    count = 0
    for i in range(len(heights)):
        if heights[i] != sort[i]:
            count += 1
    return count

# =============================================================================
# 2119. A Number After a Double Reversal

def isSameAfterReversals(self, num):
    new = int(str(num)[::-1])
    if int(str(new)[::-1]) == num:
        return True

# =============================================================================
# 2643. Row With Maximum Ones

def rowAndMaximumOnes(self, mat: List[List[int]]) -> List[int]:
    summ = []
    for i in mat:
        summ.append(sum(i))
    x = max(summ)
    return [summ.index(x), max(summ)]

# =============================================================================
# 2716. Minimize String Length

def minimizedStringLength(self, s: str) -> int:
    return len(set(s))

# =============================================================================
# 1450. Number of Students Doing Homework at a Given Time

# 1й вариант быстрее
def busyStudent(self, startTime: List[int], endTime: List[int], queryTime: int) -> int:
    count = 0
    for i in range(len(startTime)):
        if startTime[i] <= queryTime and queryTime <= endTime[i]:  # 1й вариант быстрее
            count += 1
    return count

def busyStudent(self, startTime: List[int], endTime: List[int], queryTime: int) -> int:
    count = 0
    for i in range(len(startTime)):
        if startTime[i] <= queryTime <= endTime[i]:
            count += 1
    return count

# ============================================================================
# 2341. Maximum Number of Pairs in Array

def numberOfPairs(self, nums: List[int]) -> List[int]:
    count = 0
    for i in range(len(nums)):
        x = nums[i]
        if x == None:
            continue
        if nums.count(x) >= 2:
            nums[i] = None
            index = nums.index(x)
            nums[index] = None
            count += 1
    while None in nums:
        nums.remove(None)  ## удаление элемента по значению
    return [count, len(nums)]

# =============================================================================
# 287. Find the Duplicate Number

def findDuplicate(self, nums: List[int]) -> int:
    dic = {}
    for i in nums:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    for key, val in dic.items():
        if val > 1:
            return key

def findDuplicate(self, nums: List[int]) -> int:
    for i in nums:
        if nums.count(i) > 1:
            return i  # очень медленно

# =============================================================================
# 876. Middle of the Linked List
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def middleNode(self, head: ListNode) -> ListNode:
    count = 0
    curr = head
    while curr != None:
        count += 1
        curr = curr.next
    count = count // 2 + 1

    tmp = 1
    curr = head
    while tmp < count:
        tmp += 1
        curr = curr.next
    return curr

# 2
def middleNode(self, head: ListNode) -> ListNode:
    step1 = head
    step2 = head
    while step2 != None and step2.next != None:
        step1 = step1.next
        step2 = step2.next
        if step2 != None:
            step2 = step2.next
    return step1

def middleNode(self, head: ListNode) -> ListNode:
    step1 = head
    step2 = head
    while step2 and step2.next:
        step1 = step1.next
        step2 = step2.next.next
    return step1

# =============================================================================
# 2965. Find Missing and Repeated Values

def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
    new = [0]
    for i in grid:
        new.extend(i)
    new.append(len(new))
    new = sorted(new)
    twice = None
    miss = None
    for i in range(len(new) - 1):
        if new[i + 1] - new[i] == 0:
            twice = new[i]
        elif new[i + 1] - new[i] == 2:
            miss = new[i] + 1
    return [twice, miss]

# =============================================================================
# 206. Reverse LinkedLinked List

def reverseList(self, head: ListNode) -> ListNode:
    curr = head
    prev = None
    while curr is not None:
        tail = curr.next
        curr.next = prev
        prev = curr
        curr = tail
    return prev

## recursion
def reverseList(self, head: ListNode) -> ListNode:
    if head is None:
        return None
    if head.next is None:
        return head
    tail = head.next
    head.next = None
    reversed_tail = self.reverseList(tail)
    tail.next = head
    return reversed_tail

# =============================================================================
#

def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:
    res = ListNode()
    head = res

    while list1 and list2:
        if list1.val <= list2.val:
            res.next = list1
            list1 = list1.next
        else:
            res.next = list2
            list2 = list2.next
        res = res.next

    res.next = list1 or list2
    return head.next

def mergeTwoLists(self, list1: ListNode, list2: ListNode) -> ListNode:
    curr1 = list1
    curr2 = list2

    if curr1 is None and curr2 is None:
        return None
    if curr1 is None:
        return curr2
    if curr2 is None:
        return curr1

    if list1.val <= list2.val:
        res = curr1
        curr1 = curr1.next
    else:
        res = curr2
        curr2 = list2.next
    head = res

    while curr1 and curr2:
        if curr1.val <= curr2.val:
            res.next = curr1
            res = curr1
            curr1 = curr1.next
        else:
            res.next = curr2
            res = curr2
            curr2 = curr2.next

    if curr1 is None:
        tmp = curr2
    elif curr2 is None:
        tmp = curr1
    res.next = tmp
    return head

# =============================================================================
# 2351. First Letter to Appear Twice

def repeatedCharacter(self, s: str) -> str:
    dic = {}
    for i in range(len(s)):
        if s[i] not in dic:
            dic[s[i]] = 0
        dic[s[i]] += 1
        if dic[s[i]] == 2:
            return s[i]

# =============================================================================
# 2586. Count the Number of Vowel Strings in Range

def vowelStrings(self, words: List[str], left: int, right: int) -> int:
    new = words[left:right + 1]
    vowels = ['a', 'e', 'i', 'o', 'u']
    count = 0
    for word in new:
        if word[0] in vowels and word[-1] in vowels:
            count += 1
    return count

def vowelStrings(self, words: List[str], left: int, right: int) -> int:
    vowels = 'aeiou'
    count = 0
    for i in range(left, right + 1):
        if words[i][0] in vowels and words[i][-1] in vowels:
            count += 1
    return count

def vowelStrings(self, words: List[str], left: int, right: int) -> int:
    new = words[left:right + 1]
    vowels = ['a', 'e', 'i', 'o', 'u']
    count = 0
    for word in new:
        tmp = 0
        for vowel in vowels:
            if word[0] == vowel:
                tmp += 1
            if word[-1] == vowel:
                tmp += 1
            if tmp == 2:
                break
        if tmp == 2:
            count += 1
        else:
            tmp = 0
    return count

# =============================================================================
# 79. Word Search

## нужно куча всего:
# доска  board
# занятые буквы used
# оставшаяся часть слова word
# текущее положение row, column,

def search_letter(i_row, i_column, board, word, used):
    if word == "":
        return True
    if len(word) == 1 and board[i_row][i_column] == word[0] and (i_row, i_column) not in used:
        return True
    if i_column < len(board[i_row]) - 1:  # вправо
        if board[i_row][i_column + 1] == word[0] and (i_row, i_column + 1) not in used:
            used.add((i_row, i_column + 1))
            found = search_letter(i_row, i_column + 1, board, word[1:], used)
            if found:
                return True
            used.remove((i_row, i_column + 1))

    if i_row < len(board) - 1:  # вниз
        if board[i_row + 1][i_column] == word[0] and (i_row + 1, i_column) not in used:
            used.add((i_row + 1, i_column))
            found = search_letter(i_row + 1, i_column, board, word[1:], used)
            if found:
                return True
            used.remove((i_row + 1, i_column))

    if i_column > 0:  # влево
        if board[i_row][i_column - 1] == word[0] and (i_row, i_column - 1) not in used:
            used.add((i_row, i_column - 1))
            found = search_letter(i_row, i_column - 1, board, word[1:], used)
            if found:
                return True
            used.remove((i_row, i_column - 1))

    if i_row > 0:  # вверх
        if board[i_row - 1][i_column] == word[0] and (i_row - 1, i_column) not in used:
            used.add((i_row - 1, i_column))
            found = search_letter(i_row - 1, i_column, board, word[1:], used)
            if found:
                return True
            used.remove((i_row - 1, i_column))

    return False

def exist(self, board: List[List[str]], word: str) -> bool:
    for row in range(len(board)):
        for column in range(len(board[row])):
            if search_letter(row, column, board, word[::-1], set()):
                return True
    return False

# =============================================================================
# 1614. Maximum Nesting Depth of the Parentheses

def maxDepth(self, s: str) -> int:
    count = 0
    left = 0
    right = 0
    for i in range(len(s)):
        if s[i] == ')':
            right += 1
        if s[i] == '(':
            left += 1
        depth = left - right
        if depth > count:
            count = depth
    return count

def maxDepth(self, s: str) -> int:
    count = 0
    for i in range(len(s)):
        if s[i] == ')':
            left = 0
            right = 0
            index = 0
            while index < i:
                if s[index] == '(':
                    left += 1
                elif s[index] == ')':
                    right += 1
                index += 1
            depth = left - right
            if depth > count:
                count = depth
    return count

# =============================================================================
# 1544. Make The String Great

def makeGood(self, s: str) -> str:
    i = 1
    s = list(s)
    while i < len(s):
        if ord(s[i - 1]) - ord(s[i]) == 32 or ord(s[i]) - ord(s[i - 1]) == 32:  # перевод буквы в код
            s.pop(i)
            s.pop(i - 1)
            i = 1
        else:
            i += 1
    return ''.join(s)

# =============================================================================
# 1249. Minimum Remove to Make Valid Parentheses

def minRemoveToMakeValid(self, s: str) -> str:
    stack = []
    i = 0
    s = list(s)
    while i < len(s):
        if s[i] == '(':
            stack.append(i)
        if s[i] == ')':
            if stack:
                stack.pop()
            else:
                s.pop(i)
                i -= 1
        i += 1
    while stack:
        i = stack.pop()
        s.pop(i)
    return ''.join(s)

def minRemoveToMakeValid(self, s: str) -> str:
    stack = []
    i = 0
    s = list(s)
    while i < len(s):
        if s[i] == '(':
            stack.append((s[i], i))
        if s[i] == ')':
            if stack:
                stack.pop()
            else:
                s.pop(i)
                i -= 1
        i += 1
    while stack:
        brackets, i = stack.pop()
        s.pop(i)
    return ''.join(s)

# =============================================================================
# 678. Valid Parenthesis String

def checkValidString(self, s: str) -> bool:
    stack = []
    star = []
    i = 0
    s = list(s)
    while i < len(s):
        if s[i] == '*':
            star.append(i)
        elif s[i] == '(':
            stack.append(i)
        elif s[i] == ')':
            if stack:
                stack.pop()
            else:
                if star:
                    star.pop()
                else:
                    return False
        i += 1
    while stack and star:
        for j in star:
            if stack[0] < j:
                stack.pop(0)  ## удадение по индексу
                star.remove(j)  ## удадение по значению
                break
        else:
            return False
    return len(stack) == 0

# =============================================================================
# 1700. Number of Students Unable to Eat Lunch

def countStudents(self, students: List[int], sandwiches: List[int]) -> int:
    while sandwiches and sandwiches[0] in students:
        if students[0] == sandwiches[0]:
            students.pop(0)
            sandwiches.pop(0)
        else:
            x = students.pop(0)
            students.append(x)
    return len(sandwiches)

def countStudents(self, students: List[int], sandwiches: List[int]) -> int:
    while sandwiches and sandwiches[0] in students:
        students.remove(sandwiches[0])
        sandwiches.pop(0)
    return len(sandwiches)

def countStudents(self, students: List[int], sandwiches: List[int]) -> int:
    count = 0
    i = 0
    while i < len(students):
        if students[i] == sandwiches[i]:
            students.pop(i)
            sandwiches.pop(i)
            i -= 1
        elif students[i] != sandwiches[i] and sandwiches[i] in students:
            x = students.pop(i)
            students.append(x)
            i -= 1
        elif sandwiches[i] not in students:
            return len(sandwiches)
        i += 1
    return 0

# =============================================================================
# 2073. Time Needed to Buy Tickets

def timeRequiredToBuy(self, tickets: List[int], k: int) -> int:
    res = 0
    i = 0
    while tickets[k] > 0 and i < len(tickets):
        if tickets[i] != 0:
            tickets[i] = tickets[i] - 1
            res += 1
        i += 1
        if i == len(tickets):
            i = 0
    return res

# =============================================================================
# 950. Reveal Cards In Increasing Order

def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
    new = sorted(deck)
    res = []
    while new:
        res.insert(0, new[-1])
        new.pop(-1)
        if new:
            x = res.pop(-1)
            res.insert(0, x)
    return res

# =============================================================================
# 402. Remove K Digits

def removeKdigits(self, num: str, k: int) -> str:
    stack = []
    s = []  # new int num
    for i in num:
        x = int(i)
        s.append(x)

    for elem in num:
        if not stack:
            stack.append(elem)
            continue
        while stack and stack[-1] > elem and k > 0:
            stack.pop()
            k -= 1
        stack.append(elem)

    while k > 0:
        stack.pop()
        k -= 1

    res = ''
    for i in stack:
        res += str(i)
    res = res.lstrip("0")
    if res == '':
        return '0'
    return res

# =============================================================================
# 2255. Count Prefixes of a Given String

def countPrefixes(self, words: List[str], s: str) -> int:
    count = 0
    for i in words:
        x = len(i)
        if i == s[:x]:
            count += 1
    return count

def countPrefixes(self, words: List[str], s: str) -> int:
    count = 0
    for i in words:
        x = len(i)
        if s.startswith(i):  # начинается с
            count += 1
    return count

# =============================================================================
# 42. Trapping Rain Water

def trap(self, height: List[int]) -> int:
    water = 0
    for i in range(1, len(height) - 1):
        max_left = max(height[:i])
        max_right = max(height[i + 1:])
        x = min(max_left, max_right) - height[i]
        if x > 0:
            water += x
    return water

# =============================================================================
# 1935. Maximum Number of Words You Can Type

def canBeTypedWords(self, text: str, brokenLetters: str) -> int:
    count = 0
    text = text.split()
    for word in text:
        nobroken = True
        for letter in brokenLetters:
            if letter in word:
                nobroken = False
                break
        if nobroken:
            count += 1
    return count

def canBeTypedWords(self, text: str, brokenLetters: str) -> int:
    text = text.split()
    count = len(text)
    for word in text:
        for letter in brokenLetters:
            if letter in word:
                count -= 1
                break
    return count

# =============================================================================
# 2278. Percentage of Letter in String

def percentageLetter(self, s: str, letter: str) -> int:
    count = 0
    for i in s:
        if i == letter:
            count += 1
    return int(count / len(s) * 100)

# =============================================================================
# 1. Two Sum

def twoSum(self, nums: List[int], target: int) -> List[int]:
    for i in range(len(nums) - 1):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]

# =============================================================================
# 26. Remove Duplicates from Sorted Array

def removeDuplicates(self, nums):
    i = 0
    while i < len(nums):
        if nums.count(nums[i]) > 1:
            nums.pop(i)
        else:
            i += 1
    return len(nums)

# =============================================================================
# 28. Find the Index of the First Occurrence in a String

def strStr(self, haystack: str, needle: str) -> int:
    if needle in haystack:
        return haystack.index(needle)
    return - 1

# =============================================================================
# 14. Longest Common Prefix

def longestCommonPrefix(self, strs: List[str]) -> str:
    res = ''
    for j in range(len(strs[0])):  ## для каждого индекса буквы
        prefix = strs[0][:j + 1]
        for word in strs:  ## в каждом слове
            if not word.startswith(prefix):
                return res
        res = prefix
    return res

def longestCommonPrefix(self, strs: List[str]) -> str:
    for j in range(len(strs[0])):  ## для каждого индекса буквы
        for word in strs:  ## в каждом слове
            if j >= len(word) or word[j] != strs[0][j]:
                return strs[0][:j]
    return strs[0]

# =============================================================================
# 268. Missing Number

def missingNumber(self, nums: List[int]) -> int:
    for i in range(len(nums) + 1):
        if i not in nums:
            return i

# =============================================================================
# 136. Single Number

def singleNumber(self, nums: List[int]) -> int:
    dic = {}
    for i in nums:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    for key, values in dic.items():
        if values == 1:
            return key

# =============================================================================
# 3110. Score of a String

def scoreOfString(self, s: str) -> int:
    res = 0
    for i in range(len(s) - 1):
        res += abs(ord(s[i]) - ord(s[i + 1]))
    return res

# =============================================================================
# 2057. Smallest Index With Equal Value

def smallestEqual(self, nums: List[int]) -> int:
    for i in range(len(nums)):
        if i % 10 == nums[i]:  ## "Mod" (или "modulus") - остаток от деления одного числа на другое - %
            return i
    return -1

# =============================================================================
# 2529. Maximum Count of Positive Integer and Negative Integer

def maximumCount(self, nums: List[int]) -> int:
    pos = 0
    neg = 0
    for i in nums:
        if i < 0:
            neg += 1
        elif i > 0:
            pos += 1
        return max(pos, neg)

# =============================================================================
# 2785. Sort Vowels in a String

def sortVowels(self, s: str) -> str:
    to_sort = []
    indexes = []
    vowels = 'aeiouAEIOU'
    for i in range(len(s)):
        if s[i] in vowels:
            to_sort.append(ord(s[i]))
            indexes.append(i)
    if len(to_sort) > 0:
        to_sort = sorted(to_sort)
        indexes = sorted(indexes)
    else:
        return s
    new = []
    for i in s:
        new.append(i)

    for i in range(len(to_sort)):
        new.pop(indexes[i])
        new.insert(indexes[i], chr(to_sort[i]))
    return ''.join(new)

# =============================================================================
# 1021. Remove Outermost Parentheses

def removeOuterParentheses(self, s: str) -> str:
    stack = []
    res = ''
    group = ''
    for i in s:
        if i == '(':
            stack.append(i)
            group += i
        else:
            stack.pop()
            group += i
            if not stack:
                res += group[1:-1]
                group = ''
    return res

# =============================================================================
# 160. Intersection of Two Linked Lists

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

def lenList(self, head):
    count = 0
    curr = head
    while curr is not None:
        count += 1
        curr = curr.next
    return count

def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
    len1 = self.lenList(headA)
    len2 = self.lenList(headB)
    if len1 < len2:
        minn = headA
        maxx = headB
    else:
        minn = headB
        maxx = headA

    count = 0
    while count < (abs(len1 - len2)):
        maxx = maxx.next
        count += 1

    curr_head1 = maxx
    curr_head2 = minn
    while curr_head1 != None:
        if curr_head1 != curr_head2:
            curr_head1 = curr_head1.next
            curr_head2 = curr_head2.next
        else:
            return curr_head1
    return None

## draft
def lenList(self, head):
    count = 0
    curr = head
    while curr is not None:
        count += 1
        curr = curr.next
    return count

def copyList(self, head):
    curr = head
    prev = None
    while curr is not None:
        new_node = ListNode(curr.val)
        if prev != None:
            prev.next = new_node
        else:
            new_head = new_node
        prev = new_node
        curr = curr.next
    return new_head

def reverseList(self, head):
    curr = head
    prev = None
    while curr is not None:
        tail = curr.next
        curr.next = prev
        prev = curr
        curr = tail
    return prev

def old():
    rev_head1 = self.reverseList(self.copyList(headA))
    rev_head2 = self.reverseList(self.copyList(headB))

    curr_head1 = rev_head1
    curr_head2 = rev_head2
    while curr_head1 != None or curr_head2 != None:
        if curr_head1.val == curr_head2.val:
            curr_head1 = curr_head1.next
            curr_head2 = curr_head2.next
        else:
            return self.reverseList(curr_head1)
    return None

# =============================================================================
# 1710. Maximum Units on a Truck

def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
    count = 0
    boxTypes = sorted(boxTypes, key=lambda x: x[1],
                      reverse=True)  ## сортировка по 2му элементу (индексу 1 листа в листе)

    for box, items in boxTypes:
        if box <= truckSize:
            count += items * box
            truckSize -= box
        elif truckSize > 0:
            count += truckSize * items
            break
    return count

# =============================================================================
# 2496. Maximum Value of a String in an Array

def maximumValue(self, strs: List[str]) -> int:
    res = 0
    for i in strs:
        if i.isdigit():  ## проверяется, является ли строка целым числом (стринг = число?)
            i = int(i)
            if i > res:
                res = i
        else:
            if len(i) > res:
                res = len(i)
    return res

def maximumValue(self, strs: List[str]) -> int:
    res = 0
    for i in strs:
        if i.isdigit():
            res = max(int(i), res)
        else:
            res = max(len(i), res)
    return res

# =============================================================================
# 881. Boats to Save People

def numRescueBoats(self, people: List[int], limit: int) -> int:
    people = sorted(people, reverse=True)
    left = 0
    right = len(people) - 1
    boats = 0
    while left <= right:
        if people[left] + people[right] <= limit:
            right -= 1
        left += 1
        boats += 1
    return boats

def numRescueBoats(self, people: List[int], limit: int) -> int:
    people = sorted(people, reverse=True)
    weight = 0
    boats = 0
    while len(people) > 0:
        if people[0] == limit:
            boats += 1
            people.pop(0)
        elif people[0] < limit:  # если может уместиться 2е
            weight += people[0]
            people.pop(0)
            second = limit - weight
            if len(people) > 0 and people[-1] <= second:
                people.pop()
            weight = 0
            boats += 1
    return boats

# =============================================================================
# 237. Delete Node in a Linked List

def deleteNode(self, node):
    """
    :type node: ListNode
    :rtype: void Do not return anything, modify node in-place instead.
    """
    curr = node
    prev = None
    while curr.next != None:
        curr.val = curr.next.val
        prev = curr
        curr = curr.next
    prev.next = None

def deleteNode(self, node):
    """
    :type node: ListNode
    :rtype: void Do not return anything, modify node in-place instead.
    """
    curr = node
    while curr.next.next != None:
        curr.val = curr.next.val
        curr = curr.next
    curr.val = curr.next.val
    curr.next = None

# =============================================================================
# 1002. Find Common Characters

def dic(self, word):
    dic = {}
    for i in word:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    return dic

def commonChars(self, words: List[str]) -> List[str]:
    dics = []
    for word in words:
        dics.append(self.dic(word))

    res = dics[0]

    for i in dics[1:]:
        for key in res:
            res[key] = min(i.get(key, 0), res[key])

    result = []
    for key, value in res.items():
        for i in range(value):
            result.append(key)

    return result

# =============================================================================
# 500. Keyboard Row

def check(self, word, row):
    for i in word:
        if i.lower() not in row:
            return False
    return True

def findWords(self, words: List[str]) -> List[str]:
    row1 = set("qwertyuiop")
    row2 = set("asdfghjkl")
    row3 = set("zxcvbnm")
    res = []
    for word in words:
        if word[0].lower() in row1:
            if self.check(word, row1):
                res.append(word)
        elif word[0].lower() in row2:
            if self.check(word, row2):
                res.append(word)
        elif word[0].lower() in row3:
            if self.check(word, row3):
                res.append(word)
    return res

# =============================================================================
# 657. Robot Return to Origin

def judgeCircle(self, moves: str) -> bool:
    cur = [0, 0]
    for i in moves:
        if i == "R":
            cur[0] += 1
        elif i == "L":
            cur[0] -= 1
        elif i == "U":
            cur[1] += 1
        elif i == "D":
            cur[1] -= 1
    return cur == [0, 0]

# =============================================================================
# 2475. Number of Unequal Triplets in Array

def unequalTriplets(self, nums: List[int]) -> int:
    count = 0
    for i in range(len(nums) - 2):
        for j in range(i, len(nums) - 1):
            for k in range(j, len(nums)):
                if nums[i] != nums[j] and nums[i] != nums[k] and nums[j] != nums[k]:
                    count += 1
    return count

# =============================================================================
# 1299. Replace Elements with Greatest Element on Right Side

def replaceElements(self, arr: List[int]) -> List[int]:
    maxx = -1
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] > maxx:
            tmp = arr[i]
            arr[i] = maxx
            maxx = tmp
        else:
            arr[i] = maxx
    return arr

# =============================================================================
# 506. Relative Ranks

def findRelativeRanks(self, score: List[int]) -> List[str]:
    new = sorted(score, reverse=True)
    for i in range(len(score)):
        if score[i] == new[0]:
            score[i] = "Gold Medal"
        elif score[i] == new[1]:
            score[i] = "Silver Medal"
        elif score[i] == new[2]:
            score[i] = "Bronze Medal"
        else:
            score[i] = str(new.index(score[i]) + 1)
    return score

# =============================================================================
# 2085. Count Common Words With One Occurrence

def dic(self, words):
    dic = {}
    for i in words:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1

    remove_keys = []

    for key, val in dic.items():
        if val > 1:
            remove_keys.append(key)

    while remove_keys:
        key = remove_keys.pop()
        dic.pop(key)

    return dic

def countWords(self, words1: List[str], words2: List[str]) -> int:
    dic1 = self.dic(words1)
    dic2 = self.dic(words2)

    count = 0
    for key in dic2:
        if key in dic1:
            count += 1
    return count

def dic(self, words):
    dic = {}
    for i in words:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1

    uniq_word = []

    for key, val in dic.items():
        if val == 1:
            uniq_word.append(key)
    return uniq_word

def countWords(self, words1: List[str], words2: List[str]) -> int:
    uniq_word1 = self.dic(words1)
    uniq_word2 = self.dic(words2)

    count = 0
    for i in uniq_word2:
        if i in uniq_word1:
            count += 1
    return count

# =============================================================================
# 3075. Maximize Happiness of Selected Children

def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
    count = 0
    happiness = sorted(happiness, reverse=True)
    tmp = 0

    for i in range(k):
        if happiness[i] - tmp > 0:
            count += happiness[i] - tmp
            tmp += 1
        else:
            break

    return count

# =============================================================================
# 786. K-th Smallest Prime Fraction

def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
    new = []
    for i in range(len(arr) - 1):
        for j in range(i, len(arr)):
            x = arr[i] / arr[j]
            if x == 1:
                continue
            new.append([x, arr[i], arr[j]])
    new = sorted(new)
    return [new[k - 1][1], new[k - 1][2]]

# =============================================================================
# 1636. Sort Array by Increasing Frequency

def frequencySort(self, nums: List[int]) -> List[int]:
    def dic(nums):
        dic = {}
        for i in nums:
            if i not in dic:
                dic[i] = 0
            dic[i] += 1
        return dic

    dic_nums = dic(nums)
    res = []
    pair = []

    for key, freq in dic_nums.items():
        pair.append([freq, key])

    pair = sorted(pair, key=lambda x: x[1], reverse=True)
    pair = sorted(pair, key=lambda x: x[0])

    for i in pair:
        res.extend([i[1]] * i[0])
    return res

def dic(self, nums):
    dic = {}
    for i in nums:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    return dic

def frequencySort(self, nums: List[int]) -> List[int]:
    dic_nums = self.dic(nums)

    nums.sort(reverse=True)
    nums.sort(key=lambda x: dic_nums[x])

    return nums

# =============================================================================
# 1637. Widest Vertical Area Between Two Points Containing No Points

def maxWidthOfVerticalArea(self, points: List[List[int]]) -> int:
    points = sorted(points, key=lambda x: x[0])  # сортировка по 1му элементу листа в листе
    widest = 0
    for i in range(len(points) - 1):
        diff = points[i + 1][0] - points[i][0]
        if diff > widest:
            widest = diff
    return widest

def maxWidthOfVerticalArea(self, points: List[List[int]]) -> int:
    points.sort()
    widest = 0
    for i in range(len(points) - 1):
        diff = points[i + 1][0] - points[i][0]
        if diff > widest:
            widest = diff
    return widest

# =============================================================================
# 2373. Largest Local Values in a Matrix

def largestLocal(self, grid: List[List[int]]) -> List[List[int]]:
    length = len(grid) - 1
    res = []
    for row in range(1, length):
        rows_max = []
        for column in range(1, length):
            maxx = max(grid[row][column], grid[row][column + 1], grid[row][column - 1], grid[row - 1][column],
                       grid[row - 1][column + 1], grid[row - 1][column - 1], grid[row + 1][column],
                       grid[row + 1][column + 1], grid[row + 1][column - 1])
            rows_max.append(maxx)
        res.append(rows_max)
    return res

# =============================================================================
# 3065. Minimum Operations to Exceed Threshold Value I

def minOperations(self, nums: List[int], k: int) -> int:
    nums = sorted(nums)
    steps = 0
    for i in range(len(nums)):
        if nums[i] < k:
            steps += 1
        else:
            break
    return steps

def minOperations(self, nums: List[int], k: int) -> int:
    steps = 0
    for i in range(len(nums)):
        if nums[i] < k:
            steps += 1
    return steps

def minOperations(self, nums: List[int], k: int) -> int:
    nums = sorted(nums)
    if k in nums:
        i = nums.index(k)
        return len(nums[:i])
    else:
        steps = 0
        for i in range(len(nums)):
            if nums[i] < k:
                steps += 1
            else:
                break
        return steps

# =============================================================================
# 3131. Find the Integer Added to Array I

def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
    nums1 = sorted(nums1)
    nums2 = sorted(nums2)
    return nums2[0] - nums1[0]

def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
    return min(nums2) - min(nums1)

def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
    return (sum(nums2) - sum(nums1)) // len(nums1)

# =============================================================================
# 2037. Minimum Number of Moves to Seat Everyone

def minMovesToSeat(self, seats: List[int], students: List[int]) -> int:
    seats = sorted(seats)
    students = sorted(students)
    count = 0
    for i in range(len(seats)):
        if seats[i] == students[i]:
            continue
        else:
            count += abs(seats[i] - students[i])
    return count

def minMovesToSeat(self, seats: List[int], students: List[int]) -> int:
    seats = sorted(seats)
    students = sorted(students)
    count = 0
    for i in range(len(seats)):
        count += abs(seats[i] - students[i])
    return count

# =============================================================================
# 2108. Find First Palindromic String in the Array

def firstPalindrome(self, words: List[str]) -> str:
    res = ''
    for i in words:
        if i == i[::-1]:
            return i
    return res

# =============================================================================
# 2960. Count Tested Devices After Test Operations

def countTestedDevices(self, batteryPercentages: List[int]) -> int:
    res = 0
    step = 0
    for i in batteryPercentages:
        if i - step > 0:
            res += 1
            step += 1
    return res

# =============================================================================
# 1827. Minimum Operations to Make the Array Increasing

def minOperations(self, nums: List[int]) -> int:
    count = 0
    tmp = nums[0]
    for i in range(1, len(nums)):
        if tmp >= nums[i]:
            x = tmp - nums[i] + 1
            tmp += 1
            count += x
        else:
            tmp = nums[i]
    return count

# =============================================================================
# 66. Plus One

def plusOne(self, digits: List[int]) -> List[int]:
    for i in range(len(digits) - 1, -1, -1):
        if digits[i] == 9:
            digits[i] = 0
        else:
            digits[i] += 1
            return digits
    return [1] + digits

def plusOne(self, digits: List[int]) -> List[int]:
    new = ''
    for i in digits:
        new += str(i)
    new = int(new) + 1
    new = str(new)
    res = []
    for i in new:
        res.append(int(i))
    return res

# =============================================================================
# 78. Subsets

def subsets(self, nums: List[int]) -> List[List[int]]:
    res = []
    for i in range(len(nums) + 1):  # длина
        for sub in combinations(nums, i):
            res.append(sub)
    return res

# =============================================================================
# 2706. Buy Two Chocolates

def buyChoco(self, prices: List[int], money: int) -> int:
    prices = sorted(prices)
    summ = prices[0] + prices[1]
    if summ > money:
        return money
    else:
        return money - summ

# =============================================================================
# 1046. Last Stone Weight

def lastStoneWeight(self, stones: List[int]) -> int:
    stones = sorted(stones, reverse=True)
    while len(stones) > 2:
        if stones[0] - stones[1] > 0:
            stones[0] -= stones[1]
            stones.pop(1)
            stones = sorted(stones, reverse=True)
        elif stones[0] - stones[1] == 0:
            stones.pop(1)
            stones.pop(0)
    if len(stones) == 2:
        return abs(stones[0] - stones[1])
    if len(stones) == 1:
        return stones[0]
    return 0

# =============================================================================
# 509. Fibonacci Number (recursion)

def fib(self, n: int) -> int:
    if n == 0:
        return 0
    if n == 1:
        return 1
    return self.fib(n - 1) + self.fib(n - 2)

@cache  # декоратор, сохраняет в словарь все посчитанные значения (работает намного быстрее, чем без нее)
def fib(self, n: int) -> int:
    if n == 0:
        return 0
    if n == 1:
        return 1
    return self.fib(n - 1) + self.fib(n - 2)

# =============================================================================
# 3136. Valid Word

def isValid(self, word: str) -> bool:
    vowel = 'aeiou'
    consonant = 'bcdfghjklmnpqrstvwxyz'
    digits = '0123456789'
    vow = False
    cons = False
    dig = False
    if len(word) < 3:
        return False

    word = word.lower()
    for i in word:
        if i in vowel:
            vow = True
        elif i in consonant:
            cons = True
        elif i in digits:
            dig = True
        else:
            return False
    return vow and cons

# =============================================================================
# 434. Number of Segments in a String

def countSegments(self, s: str) -> int:
    return len(s.split())

def countSegments(self, s: str) -> int:
    count = 0
    letter = False
    for i in s:
        if i == " " and letter:
            count += 1
            letter = False
        elif i != " ":
            letter = True
    if letter:
        count += 1
    return count

def countSegments(self, s: str) -> int:
    count = 0
    if len(s) == 0:
        return 0
    elif s[-1] != " ":
        count += 1

    stack = []
    for i in s:
        if i == " " and stack:
            count += 1
            stack = []
        elif i != " ":
            stack.append(i)
    return count

# =============================================================================
# 1608. Special Array With X Elements Greater Than or Equal X

def specialArray(self, nums: List[int]) -> int:
    nums = sorted(nums)
    count = 0
    minn = min(nums)
    maxx = max(nums)
    for i in range(1, maxx + 1):
        for j in nums:
            if j >= i:
                count += 1
        if count == i:
            return count
        else:
            count = 0
    return -1

# =============================================================================
# 2423. Remove Letter To Equalize Frequency

def counter(self, word):
    dic = {}
    for i in word:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    return dic

def check(self, dic):
    uniq = None
    for key, val in dic.items():
        if uniq is None:
            uniq = val
        else:
            if uniq != val:
                return False
    return True

def equalFrequency(self, word: str) -> bool:
    for i in range(len(word)):
        new = word[:i] + word[i + 1:]
        if self.check(self.counter(new)):
            return True
    return False

# =============================================================================
# 2591. Distribute Money to Maximum Children

def distMoney(self, money: int, children: int) -> int:
    child = [0] * children

    if money < children:
        return -1

    for i in range(children):  # выдали по 1
        if money > 0:
            money -= 1
            child[i] += 1

    for i in range(children):  # выдали по 1
        if money >= 7:
            money -= 7
            child[i] += 7

    if money > 0:
        for i in range(children - 1, -1, -1):
            if child[i] + money != 4:
                child[i] += money
                money = 0
            else:
                child[i] += money - 1
                money = 1
    return child.count(8)

# =============================================================================
# 1909. Remove One Element to Make the Array Strictly Increasing

def canBeIncreasing(self, nums: List[int]) -> bool:
    for i in range(len(nums)):
        new = nums[:i] + nums[i + 1:]
        increasing = True
        for j in range(1, len(new)):
            if new[j - 1] >= new[j]:
                increasing = False
                break
        if increasing:
            return True
    return increasing

# =============================================================================
# 2486. Append Characters to String to Make Subsequence

def appendCharacters(self, s: str, t: str) -> int:
    i_s = 0
    i_t = 0
    while i_s < len(s) and i_t < len(t):
        if s[i_s] == t[i_t]:
            i_t += 1
        i_s += 1
    return len(t[i_t:])

# =============================================================================
# 409. Longest Palindrome

def count_letters(self, word):
    dic = {}
    for i in word:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    return dic

def longestPalindrome(self, s: str) -> int:
    res = 0
    odd = False
    dic = self.count_letters(s)
    for key, val in dic.items():
        if val % 2 != 0:
            odd = True
            res -= 1
        res += val
    if odd:
        res += 1
    return res

# =============================================================================
# 846. Hand of Straights

def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
    if len(hand) % groupSize != 0:
        return False
    hand.sort()
    i = 0
    group = []
    while i < len(hand):
        if len(group) == 0:
            group.append(hand.pop(i))
        if i < len(hand) and len(group) < groupSize:
            if hand[i] == group[-1]:
                if len(hand) == 1:
                    return False
                i += 1
            elif hand[i] - group[-1] == 1:
                group.append(hand.pop(i))
            elif hand[i] - group[-1] > 1:
                return False
        if len(group) == groupSize:
            group = []
            i = 0
    if i == len(hand) and group:
        return False
    return True

# =============================================================================
# 648. Replace Words

def replaceWords(self, dictionary: List[str], sentence: str) -> str:
    new = sentence.split()
    for root in dictionary:
        for word in new:
            if word.startswith(root):
                new[new.index(word)] = root
    return ' '.join(new)

# =============================================================================
# 605. Can Place Flowers

def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
    count = 0
    for i in range(len(flowerbed)):
        if flowerbed[i] == 0:
            empty_left = (i == 0) or (flowerbed[i - 1] == 0)
            empty_right = (i == len(flowerbed) - 1) or (flowerbed[i + 1] == 0)

            if empty_left and empty_right:
                flowerbed[i] = 1
                count += 1
                if count >= n:
                    return True
    return count >= n

# =============================================================================
# 925. Long Pressed Name

def isLongPressedName(self, name: str, typed: str) -> bool:
    if name[0] != typed[0]:
        return False

    i = 0
    for j in typed:
        if i < len(name) and name[i] == j:
            i += 1
        elif name[i - 1] == j:
            continue
        else:
            return False
    return i == len(name)

# =============================================================================
# 1122. Relative Sort Array

def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
    arr1.sort()
    arr1.sort(key=lambda x: arr2.index(x) if x in arr2 else 1000)
    return arr1

# =============================================================================
# 75. Sort Colors

def sortColors(self, nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    nums.sort()

# =============================================================================
# 945. Minimum Increment to Make Array Unique

def minIncrementForUnique(self, nums: List[int]) -> int:
    count = 0
    nums.sort()
    for i in range(1, len(nums)):
        if nums[i] <= nums[i - 1]:
            diff = (nums[i - 1] - nums[i])
            nums[i] = nums[i] + diff + 1
            count = count + diff + 1
    return count

# =============================================================================
# 414. Third Maximum Number

def thirdMax(self, nums: List[int]) -> int:
    new = list(set(nums))
    new.sort(reverse=True)
    if len(new) > 2:
        return new[2]
    return new[0]

# =============================================================================
# 1346. Check If N and Its Double Exist

def checkIfExist(self, arr: List[int]) -> bool:
    if arr.count(0) == 1:
        arr.remove(0)
    for i in arr:
        if i * 2 in arr:
            return True
    return False

# =============================================================================
# 2047. Number of Valid Words in a Sentence

def countValidWords(self, sentence: str) -> int:
    count = 0
    words = sentence.split()
    digits = '0123456789'

    for word in words:
        valid = False
        for letter in word:
            if letter in digits:
                valid = False
                break
            elif letter == "-":
                if word.count(letter) > 1:
                    valid = False
                    break
                elif letter not in word[1:-1]:
                    valid = False
                    break
            elif letter == "!" or letter == "." or letter == ",":
                if letter in word[:-1]:
                    valid = False
                    break
                elif len(word) > 1 and word[-2] == "-":
                    valid = False
                    break
                else:
                    valid = True
            else:
                valid = True

        if valid:
            count += 1
    return count

# =============================================================================
# 263. Ugly Number

def isUgly(self, n: int) -> bool:
    while n > 1:
        if n % 2 == 0:
            n = n / 2
        elif n % 3 == 0:
            n = n / 3
        elif n % 5 == 0:
            n = n / 5
        else:
            break
    return n == 1

# =============================================================================
# 2259. Remove Digit From Number to Maximize Result

def removeDigit(self, number: str, digit: str) -> str:
    new = []
    for i in range(len(number)):
        if number[i] == digit:
            new.append(int(number[:i] + number[i + 1:]))
            number[:i] + number[i + 1:]
    return str(max(new))

# =============================================================================
# 2553. Separate the Digits in an Array

def separateDigits(self, nums: List[int]) -> List[int]:
    res = []
    for i in nums:
        num = str(i)
        for j in num:
            res.append(int(j))
    return res

# =============================================================================
# 1550. Three Consecutive Odds

def threeConsecutiveOdds(self, arr: List[int]) -> bool:
    count = 0
    for i in arr:
        if i % 2 != 0:
            count += 1
            if count == 3:
                return True
        else:
            count = 0
    return False

def threeConsecutiveOdds(self, arr: List[int]) -> bool:
    count = 0
    for i in arr:
        if i % 2 != 0:
            count += 1
        else:
            count = 0
        if count == 3:
            return True
    return False

# =============================================================================
# 350. Intersection of Two Arrays II

def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
    res = []
    nums1.sort()
    nums2.sort()
    i = 0
    j = 0
    while i < len(nums1) and j < len(nums2):
        if nums1[i] == nums2[j]:
            res.append(nums1[i])
            i += 1
            j += 1
        elif nums1[i] < nums2[j]:
            i += 1
        elif nums1[i] > nums2[j]:
            j += 1
    return res

# =============================================================================
# 1509. Minimum Difference Between Largest and Smallest Value in Three Moves

def minDifference(self, nums: List[int]) -> int:
    if len(nums) < 5:
        return 0

    nums.sort()
    min_diff = float("Inf")
    # inf = float("Inf") Бесконечность (Infinity),
    # neg_inf = float("-Inf") - Отрицательная бесконечность (-Infinity),
    # nan = float("NaN") - Не число (NaN - Not a Number)

    for i in range(4):
        min_diff = min(min_diff, nums[i - 4] - nums[i])
    return min_diff

def minDifference(self, nums: List[int]) -> int:
    if len(nums) < 5:
        return 0

    nums.sort()
    min_diff = 10000000000
    for i in range(4):
        min_diff = min(min_diff, nums[i - 4] - nums[i])
    return min_diff

def minDifference(self, nums: List[int]) -> int:
    if len(nums) < 5:
        return 0

    nums.sort()
    return min(
        nums[-4] - nums[0],
        nums[-3] - nums[1],
        nums[-2] - nums[2],
        nums[-1] - nums[3]
    )

# =============================================================================
# 2582. Pass the Pillow

def passThePillow(self, n: int, time: int) -> int:
    i = 0
    while i < n and time > 0:
        i += 1
        time -= 1
    if i == n - 1:
        while i > 0 and time > 0:
            i -= 1
            time -= 1
    return i + 1

# =============================================================================
# 1518. Water Bottles

def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
    drink = numBottles

    while numBottles // numExchange > 0:
        empty = numBottles // numExchange
        drink += numBottles // numExchange
        numBottles = numBottles - (numBottles // numExchange * numExchange) + empty

    return drink

# =============================================================================
#

def averageWaitingTime(self, customers: List[List[int]]) -> float:
    currentTime = 0
    totalwaitTime = 0

    for arrival, time in customers:
        if currentTime < arrival:
            currentTime = arrival
        waitTime = currentTime + time - arrival
        totalwaitTime += waitTime
        currentTime += time

    return totalwaitTime / len(customers)

# =============================================================================
# 860. Lemonade Change

def lemonadeChange(self, bills: List[int]) -> bool:
    five = 0
    ten = 0
    twenty = 0
    for i in bills:
        if i == 5:
            five += 1
        elif i == 10:
            if five != 0:
                five -= 1
                ten += 1
            else:
                return False
        elif i == 20:
            if five > 0 and ten > 0:
                five -= 1
                ten -= 1
                twenty += 1
            elif five > 2:
                five -= 3
                twenty += 1
            else:
                return False
    return True

# =============================================================================
# 624. Maximum Distance in Arrays

def maxDistance(self, arrays: List[List[int]]) -> int:
    minn = sorted(arrays, key=lambda x: x[0])
    maxx = sorted(arrays, key=lambda x: x[-1], reverse=True)
    if maxx[0] != minn[0]:
        return abs(maxx[0][-1] - minn[0][0])
    return max(abs(maxx[0][-1] - minn[1][0]), abs(maxx[1][-1] - minn[0][0]))

# =============================================================================
# 151. Reverse Words in a String

def reverseWords(self, s: str) -> str:
    x = s.split()
    res = ''
    for i in range(len(x) - 1, -1, -1):
        res += x[i]
        if i != 0:
            res += ' '
    return res

def reverseWords(self, s: str) -> str:
    x = s.split()[::-1]
    return ' '.join(x)

# =============================================================================
# 334. Increasing Triplet Subsequence

def increasingTriplet(self, nums: List[int]) -> bool:
    first = 2 ** 31
    second = 2 ** 31
    for i in nums:
        if i < first:
            first = i
        if first < i < second:
            second = i
        if i > second:
            return True
    return False

# =============================================================================
# 258. Add Digits

def addDigits(self, num: int) -> int:
    res = num
    tmp = res
    while len(str(res)) > 1:
        res = 0
        for i in str(tmp):
            res += int(i)
        tmp = res
    return res

# =============================================================================
# 171. Excel Sheet Column Number

def titleToNumber(self, columnTitle: str) -> int:
    res = 0
    pos = 0
    for i in range(len(columnTitle) - 1, -1, -1):
        res += (26 ** pos) * (ord(columnTitle[i]) - 64)
        pos += 1
    return res

# =============================================================================
# 476. Number Complement

def findComplement(self, num: int) -> int:
    x = format(num, 'b')
    ## format(14, '#b'), format(14, 'b') -->('0b1110', '1110'); f'{14:#b}', f'{14:b}' -->('0b1110', '1110')
    y = ''
    for i in str(x):
        if i == '0':
            y += '1'
        else:
            y += '0'
    res = int(y, base=2)  # из bin в число
    return res

# =============================================================================
# 67. Add Binary

def addBinary(self, a: str, b: str) -> str:
    return bin(
        int(a, 2) + int(b, 2)
    )[2:]

# =============================================================================
# 338. Counting Bits

def countBits(self, n: int) -> List[int]:
    ans = []
    for i in range(n + 1):
        ans.append(bin(i).count('1'))
    return ans

def countBits(self, n: int) -> List[int]:
    res = []
    for i in range(n+1):
        x = 0
        for j in str(format(i, 'b')):
            x += int(j)
        res.append(x)
    return res

# =============================================================================
# 2022. Convert 1D Array Into 2D Array

def construct2DArray(self, original: List[int], m: int, n: int) -> List[List[int]]:
    if n * m != len(original):
        return []
    res = []
    while m > 0:
        tmp = []
        for i in range(n):
            tmp.append(original[0])
            original.pop(0)
        res.append(tmp)
        m = m - 1
    return res

# =============================================================================
# 1945. Sum of Digits of String After Convert

def getLucky(self, s: str, k: int) -> int:
    res = ''
    for i in s:
        res += str(ord(i) - 96) ## или str(ord(x) - ord('a') + 1), ord('a') = 97

    while k > 0:
        tmp = 0
        for i in str(res):
            tmp += int(i)
        res = tmp
        k -= 1
    return res

# =============================================================================
# 88. Merge Sorted Array

# for i in range(n):
#     nums1.pop()
# nums1 += nums2
# nums1.sort()

# =============================================================================
# 3168. Minimum Number of Chairs in a Waiting Room
def minimumChairs(self, s: str) -> int:
    num = 0
    res = 0
    for i in s:
        if i == 'E':
            num += 1
            if num > res:
                res = num
        else:
            num -= 1
    return res

# =============================================================================
# 3099. Harshad Number

def sumOfTheDigitsOfHarshadNumber(self, x: int) -> int:
    num = ''
    for i in str(x):
        num += i
    summ = 0
    for i in num:
        summ += int(i)

    if x % summ == 0:
        return summ
    return -1

# =============================================================================
# 1668. Maximum Repeating Substring

def maxRepeating(self, sequence: str, word: str) -> int:
    res = 0
    tmp = 0
    x = len(word)
    i = 0
    while i < len(sequence) - x + 1:
        if sequence[i:x + i] == word:
            tmp += 1
            res = max(tmp, res)
            i += x
        elif tmp > 0:
            tmp = 0
            i -= x - 1
        else:
            i += 1
            tmp = 0
    return res

# =============================================================================
# 832. Flipping an Image

def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
    new = []
    for i in image:
        new.append(i[::-1])
    res = []
    for i in new:
        tmp = []
        for j in i:
            if j == 0:
                tmp.append(1)
            else:
                tmp.append(0)
        res.append(tmp)
    return res

# =============================================================================
# 69. Sqrt(x)

def mySqrt(self, x: int) -> int:
    return int(x ** 0.5)

# =============================================================================
# 1984. Minimum Difference Between Highest and Lowest of K Scores

def minimumDifference(self, nums: List[int], k: int) -> int:
    if len(nums) == 1:
        return 0
    nums.sort()
    res = 100000  # float('inf') - бесконечность
    for i in range(len(nums)-k+1):
        diff = nums[i+k-1] - nums[i]
        if diff < res:
            res = diff
            if diff == 0:
                return 0
    return res

# =============================================================================
# 819. Most Common Word

class Solution:
    def count(self, word):
        dic = {}
        for i in word:
            if i not in dic:
                dic[i] = 0
            dic[i] += 1
        return dic

    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        banned = set(banned)
        paragraph = paragraph.lower()
        symbols = "!?',;."
        for i in symbols:
            if i in paragraph:
                paragraph = paragraph.replace(i, ' ')
        paragraph = paragraph.split()

        x = self.count(paragraph)
        new = sorted(list(x.keys()), key=lambda y: -x[y])
        for i in new:
            if i not in banned:
                return i

# =============================================================================
# 3046. Split the Array

def count(self, word):
    dic = {}
    for i in word:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    return dic
def isPossibleToSplit(self, nums: List[int]) -> bool:
    x = self.count(nums)
    for key, val in x.items():
        if val > 2:
            return False
    return True

# =============================================================================
# 1185. Day of the Week

import datetime
class Solution:
    def dayOfTheWeek(self, day: int, month: int, year: int) -> str:
        date = datetime.datetime(year, month, day)
        x = date.weekday()  # определение дня недели по дате
        return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x]

# =============================================================================
# 202. Happy Number

def square(self, num):
    res = 0
    for digit in str(num):
        res += int(digit) ** 2
    return res

def isHappy(self, n: int) -> bool:
    seen = set([n])
    while n != 1:
        n = self.square(n)
        if n in seen:
            return False
        seen.add(n)
    return True

# =============================================================================
# 539. Minimum Time Difference

def findMinDifference(self, timePoints: List[str]) -> int:
    minutes = []
    for i in timePoints:
        x = int(i[0:2]) * 60 + int(i[3:5])
        minutes.append(x)
    minutes.sort()
    minn = 1440
    for i in range(1, len(minutes)):
        minn = min((minutes[i]-minutes[i-1]), minn)
    if len(minutes) > 1:
        x = 1440 - minutes[-1] + minutes[0]
        minn = min(minn, x)
    return minn

# =============================================================================
# 884. Uncommon Words from Two Sentences

def count(self, word):
    dic = {}
    for i in word:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    return dic
def uncommonFromSentences(self, s1: str, s2: str) -> List[str]:
    common = s1 + ' ' + s2
    res = []
    x = self.count(common.split())
    for key, val in x.items():
        if val == 1:
            res.append(key)
    return res

# =============================================================================
# 179. Largest Number

def largestNumber(self, nums: List[int]) -> str:
    new = []
    for i in nums:
        new.append(str(i))

    # new = [str(i) for i in nums]

    new.sort(reverse=True) ## неверная сортировка
    return ''.join(new)

# =============================================================================
# 1556. Thousand Separator

def thousandSeparator(self, n: int) -> str:
    n = str(n)
    res = ''
    step = 0
    for i in range((len(n))-1, -1, -1):
        res = n[i] + res
        step += 1
        if step == 3 and i != 0:
            res = '.' + res
            step = 0
    return res

# =============================================================================
# 290. Word Pattern

def wordPattern(self, pattern: str, s: str) -> bool:
    new = s.split()
    if len(new) != len(pattern):
        return False

    dic = {}
    for i in range(len(new)):
        if new[i] not in dic:
            dic[new[i]] = pattern[i]
        else:
            if dic[new[i]] != pattern[i]:
                return False
    values = list(dic.values())
    return len(values) == len(set(values))

# =============================================================================
# 1805. Number of Different Integers in a String

def numDifferentIntegers(self, word: str) -> int:
    for i in word:
        if not i.isdigit():  ## проверка, что символ не число
            word = word.replace(i, " ")
    word = word.split()
    res = set()
    for i in word:
        res.add(int(i))
    return len(res)

# =============================================================================
# 386. Lexicographical Numbers

def lexicalOrder(self, n: int) -> List[int]:
    return [int(i) for i in sorted([str(i) for i in list(range(1, n+1))])]

def lexicalOrder(self, n: int) -> List[int]:
    new = list(range(1, n+1))
    res = sorted([str(i) for i in new])
    res2 = [int(i) for i in res]  # List comprehension
    return res2

def lexicalOrder(self, n: int) -> List[int]:
    new = list(range(1, n+1))
    res = []
    for i in new:
        res.append(str(i))
    res.sort()
    res2 = []
    for i in res:
        res2.append(int(i))
    return res2

# =============================================================================
# 520. Detect Capital

def detectCapitalUse(self, word: str) -> bool:
    return word.isupper() or word.islower() or word.istitle()  # istitle() - с 1я заглавная, остальные строчные

def detectCapitalUse(self, word: str) -> bool:
    return word.isupper() or word.islower() or (word[0].isupper() and word[1:].islower())

# =============================================================================
# 415. Add Strings

import sys

def addStrings(self, num1: str, num2: str) -> str:
    sys.set_int_max_str_digits(6000)
    return str(int(num1) + int(num2))

# =============================================================================
# 2079. Watering Plants

def wateringPlants(self, plants: List[int], capacity: int) -> int:
    wat = capacity
    i = 0
    steps = 0
    while i < len(plants):
        if wat >= plants[i]:
            wat -= plants[i]
            i += 1
            if i < len(plants) and wat >= plants[i]:
                steps += 1
        else:
            wat = capacity
            steps += (i + 1) * 2 - 1
    steps += 1
    return steps

# =============================================================================
# 1910. Remove All Occurrences of a Substring

def removeOccurrences(self, s: str, part: str) -> str:
    n = len(part)
    while part in s:
        i = s.find(part)
        s = s[:i] + s[i+n:] ## удаление в стринге через срез
    return s

def removeOccurrences(self, s: str, part: str) -> str:
    while part in s:
        s = s.replace(part, "", 1) ## удаление в стринге через замену replace
    return s

# =============================================================================
# 3227. Vowels Game in a String

from collections import Counter
def doesAliceWin(self, s: str) -> bool:
    vowels = 'aeiou'
    counter = Counter(letter for letter in s if letter in vowels)  ## List comprehension с условием. считаем только гласные, вернет словарь
    count_vowels = sum(counter.values())
    return count_vowels != 0

def doesAliceWin(self, s: str) -> bool:
    vowels = 'aeiou'
    for i in s:
        if i in vowels:
            return True
    return False

# =============================================================================
# 2243. Calculate Digit Sum of a String

def digitSum(self, s: str, k: int) -> str:
    new = ''
    while len(s) > k:
        i = 0
        tmp = 0
        for num in s:
            if i < k:
                if tmp is None:
                    tmp = int(num)
                else:
                    tmp += int(num)
                i += 1
                if i == k:
                    new += str(tmp)
                    tmp = None
                    i = 0
        if tmp is not None:
            new += str(tmp)
        s = new
        new = ''
    return s

# =============================================================================
# 1561. Maximum Number of Coins You Can Get

def maxCoins(self, piles: List[int]) -> int:
    res = 0
    x = len(piles)
    piles.sort(reverse=True)
    for i in range(1, x // 3 * 2, 2):
        res += piles[i]
    return res

def maxCoins(self, piles: List[int]) -> int:
    res = 0
    piles.sort(reverse=True)
    while len(piles) > 2:
        piles.pop(0)
        res += piles[0]
        piles.pop(0)
        piles.pop(-1)
    return res

# =============================================================================
# 2433. Find The Original Array of Prefix Xor

def findArray(self, pref: List[int]) -> List[int]:
    res = [pref[0]]
    for i in range(1, len(pref)):
        res.append(pref[i-1] ^ pref[i])  # чтобы расксорить, надо заксорить еще раз. и все :)
    return res

# =============================================================================
# 3158. Find the XOR of Numbers Which Appear Twice

from collections import Counter


def duplicateNumbersXOR(self, nums: List[int]) -> int:
    res = 0
    counter = Counter(nums)
    double = [key for key, value in counter.items() if value > 1]
    if len(double) >= 1:
        res = double[0]
        for i in range(1, len(double)):
            res = double[i] ^ res
    return res

# =============================================================================
# 2390. Removing Stars From a String

def removeStars(self, s: str) -> str:
    stack = []
    for i in s:
        if i != '*':
            stack.append(i)
        else:
            stack.pop()
    return ''.join(stack)


def removeStars(self, s: str) -> str: #медленно
    while '*' in s:
        i = s.index('*')
        s = s[:i - 1] + s[i + 1:]
    return s

# =============================================================================
# 1331. Rank Transform of an Array

def arrayRankTransform(self, arr: List[int]) -> List[int]:
    uniq = set(arr)
    sorted_list = sorted(uniq)  ## метод .sort() изменяет исходник. функция sorted() не изменяет исходник, сохранять в новую переменную
    dic = {}
    for i in range(1, len(sorted_list)+1):
        dic[sorted_list[i-1]] = i
    for i in range(len(arr)):
        arr[i] = dic[arr[i]]
    return arr

# =============================================================================
#1154. Day of the Year

import datetime
class Solution:
    def dayOfYear(self, date: str) -> int:
        year = int(date[:4])
        month = int(date[5:7])
        day = int(date[8:])
        date = datetime.date(year, month, day)
        return date.timetuple().tm_yday  # Подсчет номера дня в году. даты

# =============================================================================
# 551. Student Attendance Record I

def checkRecord(self, s: str) -> bool:
    return s.count("A")<2 and "LLL" not in s

def checkRecord(self, s: str) -> bool:
    count_a = 0
    l_days = 0
    max_days = 0
    for i in s:
        if i == "A":
            count_a += 1
        if i != "L":
            l_days = 0
        elif i == "L":
            l_days += 1
            max_days = max(max_days, l_days)
    return count_a < 2 and max_days < 3

# =============================================================================
# 645. Set Mismatch

def findErrorNums(self, nums: List[int]) -> List[int]:
    x = len(nums)
    new = list(range(1, x+1))
    skip = 0
    double = 0
    for i in new:
        if i not in nums:
            skip = i
    seen = set()  # поиск дубликата/ задвоения вместо Couner (каунтером тоже можно)
    for num in nums:
        if num in seen:
            double = num
            break
        seen.add(num)
    return [double, skip]

# =============================================================================
# 1813. Sentence Similarity III

def areSentencesSimilar(self, sentence1: str, sentence2: str) -> bool:
    new1 = sentence1.split()
    new2 = sentence2.split()
    while 0 < len(new1) and 0 < len(new2) and new1[0] == new2[0]:
        new1.pop(0)
        new2.pop(0)
    while 0 < len(new1) and 0 < len(new2) and new1[-1] == new2[-1]:
        new1.pop()
        new2.pop()
    return not new1 or not new2

# =============================================================================
# 921. Minimum Add to Make Parentheses Valid

def minAddToMakeValid(self, s: str) -> int:
    stack = []
    for i in s:
        if i == "(":
            stack.append(i)
        elif i == ")":
            if stack:
                stack.pop()
    return len(stack)

# =============================================================================
# 20. Valid Parentheses

def isValid(self, s: str) -> bool:
    stack = []
    for i in s:
        if i in '({[':  # можно было черех словать решить
            stack.append(i)
        elif i == ')':
            if stack and stack[-1] == '(':
                stack.pop()
            else:
                return False
        elif i == ']':
            if stack and stack[-1] == '[':
                stack.pop()
            else:
                return False
        elif i == '}':
            if stack and stack[-1] == '{':
                stack.pop()
            else:
                return False
    return len(stack) == 0

# =============================================================================
# 3264. Final Array State After K Multiplication Operations I

def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
    i = 0
    while i < k:
        x = min(nums)
        i_min = nums.index(x)
        nums[i_min] = x * multiplier
        i += 1
    return nums


def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
    for _ in range(k):
        i_min = nums.index(min(nums))
        nums[i_min] *= multiplier
    return nums

# =============================================================================
# 2696. Minimum String Length After Removing Substrings

def minLength(self, s: str) -> int:
    while "AB" in s or "CD" in s:
        if "AB" in s:
            i = s.index("AB")
            s = s[:i] + s[i+2:]  # срезы начало включительно, конец не включителен
        if "CD" in s:
            i = s.index("CD")
            s = s[:i] + s[i+2:]
    return len(s)

# =============================================================================
# 682. Baseball Game

def calPoints(self, operations: List[str]) -> int:
    res = []
    i = 0
    while i < len(operations):
        if operations[i] == '+':
            res.append((res[-1])+(res[-2]))
        elif operations[i] == 'D':
            res.append((res[-1])*2)
        elif operations[i] == 'C':
            res.pop()
        else:
           res.append(int(operations[i]))
        i += 1
    return sum(res)

def calPoints(self, operations: List[str]) -> int:
    res = []
    for i in operations:
        if i == '+':
            res.append((res[-1])+(res[-2]))
        elif i == 'D':
            res.append((res[-1])*2)
        elif i == 'C':
            res.pop()
        else:
           res.append(int(i))
    return sum(res)

# =============================================================================
# 3174. Clear Digits

def clearDigits(self, s: str) -> str:
    i = 0
    while i < len(s):
        if s[i].isdigit():
            s = s[:i - 1] + s[i + 1:]
            i -= 1
        else:
            i += 1
    return s

# =============================================================================
# 670. Maximum Swap

def maximumSwap(self, num: int) -> int: ## неверное ренение, тк неверно поняты условия
    new_num = str(num)
    maxx = max(new_num)
    count_max = new_num.count(maxx)
    last_max = new_num.rfind(maxx)  # метод находит индекс последнего вхождения подстроки
    i = 0
    while count_max > 0:
        x = new_num[i]
        if x != maxx:
            new_num = new_num.replace(new_num[i], maxx, 1)
            new_num = new_num[:last_max] + x + new_num[last_max+1:]
        count_max -= 1
        i += 1
    return int(new_num)

# =============================================================================
# 1360. Number of Days Between Two Dates
from datetime import datetime

def daysBetweenDates(self, date1: str, date2: str) -> int:
    date_format = "%Y-%m-%d"
    d1 = datetime.strptime(date1, date_format) #переводим стринг в формат даты
    d2 = datetime.strptime(date2, date_format)

    return abs((d2 - d1).days)

# =============================================================================
# 1576. Replace All ?'s to Avoid Consecutive Repeating Characters

def modifyString(self, s: str) -> str:
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    i = 0
    j = 0
    while i < len(s):
        if s[i] == '?':
            if i == len(s) - 1 and s[i - 1] != alphabet[j]:
                s = s[:i] + alphabet[j]
                j += 1
            elif s[i - 1] != alphabet[j] and s[i + 1] != alphabet[j]:
                s = s[:i] + alphabet[j] + s[i + 1:]
                j += 1
            else:
                j += 1
                i -= 1
        if j == len(alphabet) - 1:
            j = 0
        i += 1
    return s

# =============================================================================
# 541. Reverse String II

def reverseStr(self, s: str, k: int) -> str:
    new = list(s)
    for i in range(0, len(s), k*2):
        new[i:i+k] = reversed(new[i:i+k])
    return ''.join(new)

# =============================================================================
# 2437. Number of Valid Clock Times

def match(self, curr_time, time):
    n = len(curr_time)
    for i in range(n):
        if time[i] != '?' and curr_time[i] != time[i]:
            return False
    return True

def countTime(self, time: str) -> int:
    res = 0
    hours = [f"{i:02}" for i in range(0,
                                      24)]  # Здесь :02 форматирует числа так, чтобы было всегда 2 цифры, добавляя ведущий ноль, если нужно.
    minutes = [f"{i:02}" for i in range(0, 60)]
    for hour in hours:
        for minute in minutes:
            curr_time = hour + ':' + minute
            if self.match(curr_time, time):
                res += 1
    return res

# =============================================================================
# 680. Valid Palindrome II

def checkPalindrime(self, s):
    return s == s[::-1]

def validPalindrome(self, s: str) -> bool:
    h = 0
    t = len(s) - 1
    while h < t:
        if s[h] != s[t]:
            return self.checkPalindrime(s[h + 1:t + 1]) or self.checkPalindrime(s[h:t])
        h += 1
        t -= 1
    return True

# =============================================================================
# 1957. Delete Characters to Make Fancy String

def makeFancyString(self, s: str) -> str:
    if len(s) < 3:
        return s
    i = 2
    while i < len(s):
        if s[i] == s[i-1] == s[i-2]:
            s = s[:i] + s[i+1:]
        else:
            i += 1
    return s

# =============================================================================
# 2490. Circular Sentence

def isCircularSentence(self, sentence: str) -> bool:
    new = sentence.split()
    letter = new[-1][-1]
    for i in new:
        if i[0] == letter:
            letter = i[-1]
        else:
            return False
    return True

# =============================================================================
# 796. Rotate String

def rotateString(self, s: str, goal: str) -> bool:
    for i in range(len(s)):
        if s[i:] + s[:i] == goal:
            return True
    return False

def rotateString(self, s: str, goal: str) -> bool:
    if len(s) != len(goal):
        return False
    return goal in s + s

# =============================================================================
# 3194. Minimum Average of Smallest and Largest Elements

def minimumAverage(self, nums: List[int]) -> float:
    averages = []
    while len(nums) > 1:
        maxx = max(nums)
        minn = min(nums)
        summ = (maxx + minn) / 2
        nums.remove(maxx)
        nums.remove(minn)
        averages.append(summ)
    return round(min(averages), 2)  # округление float до 2 символов после запятой

# =============================================================================
# 2248. Intersection of Multiple Arrays

def count(self, nums):
    dic = {}
    for i in nums:
        if i not in dic:
            dic[i] = 0
        dic[i] += 1
    return dic
def intersection(self, nums: List[List[int]]) -> List[int]:
    new = [i for sub in nums for i in sub]
    dic = self.count(new)
    res = []
    for key, val in dic.items():
        if val == len(nums):
            res.append(key)
    return sorted(res)

# =============================================================================
# 7. Reverse Integer

def reverse(self, x: int) -> int:
    new = str(x)
    negative = False
    if "-" in new:
        negative = True
        new = new[1:]
    new = new[::-1]
    res = int(new)
    if res > 2 ** 31:
        return 0
    if negative:
        res = res * -1
    return res

# =============================================================================
#

# =============================================================================
#

# =============================================================================
#

# =============================================================================
#

# =============================================================================
#

# =============================================================================
#

# =============================================================================
#

# =============================================================================
#



# =============================================================================
#


