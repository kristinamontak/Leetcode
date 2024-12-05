-- =============================================================================
-- 1741. Find Total Time Spent by Each Employee


select
event_day as day,
emp_id,
sum(out_time-in_time) as total_time
from Employees
group by event_day, emp_id


-- =============================================================================
-- 1693. Daily Leads and Partners

select date_id, make_name, count(distinct lead_id) as unique_leads, count(distinct partner_id) as unique_partners
from DailySales
group by date_id, make_name

-- =============================================================================
-- 2356. Number of Unique Subjects Taught by Each Teacher


select teacher_id, count(distinct subject_id) as cnt
from Teacher
group by teacher_id

-- =============================================================================
-- 1795. Rearrange Products Table

-- 1) транспонирование столбцов в строки через unpivot
select product_id, store, price
from Products
unpivot 
(
price
for store
in (store1, store2, store3)
)
as pivoted_table
order by product_id;


-- 2) объединение через union, присвоение значения новому столбцу

select product_id, 'store1' as store, store1 as price
from Products
where store1 is not null
union
select product_id, 'store_2' as store, store2 as price
from Products
where store2 is not null
union
select product_id, 'store3' as store, store3 as price
from Products
where store3 is not null

-- =============================================================================
-- 1683. Invalid Tweets
-- len() получает количество символов


select tweet_id from Tweets
where len(content) > 15;

-- =============================================================================
-- 1587. Bank Account Summary II

select Users.name, sum(Transactions.amount) as balance 
from Users join Transactions
on Users.account = Transactions.account
group by Users.name
having sum(Transactions.amount) > 10000;


select name, balance
from (
  select account, sum(amount) as balance
  from Transactions
  group by account
) as account_balances
join Users on account_balances.account = Users.account
where balance > 10000;

-- =============================================================================
-- 627. Swap Salary

update Salary
set sex = case
when sex = 'm' then 'f'
when sex = 'f' then 'm'
end;


-- =============================================================================
-- 1378. Replace Employee ID With The Unique Identifier

select EmployeeUNI.unique_id, Employees.name from EmployeeUNI
right join Employees
on EmployeeUNI.id = Employees.id;

select EmployeeUNI.unique_id, Employees.name from Employees
left join EmployeeUNI
on EmployeeUNI.id = Employees.id;

-- =============================================================================
-- 1068. Product Sales Analysis I

select Product.product_name, Sales.year, Sales.price
from Sales
join Product
on Sales.product_id = Product.product_id

-- =============================================================================
-- 1890. The Latest Login in 2020

select user_id, max(time_stamp) as last_stamp
from Logins
where year(time_stamp) = '2020'
group by user_id


select user_id, max(time_stamp) as last_stamp
from Logins
where time_stamp between '2020-01-01' and '2020-12-31 23:59:59'
group by user_id

-- =============================================================================
-- 1484. Group Sold Products By The Date

mysql
select sell_date, count(distinct product) as num_sold,
group_concat(distinct product order by product) as products
from Activities
group by sell_date

select sell_date, count(distinct product) as num_sold,
group_concat(distinct product order by product separator ',') as products
from Activities
group by sell_date

msSql
select sell_date, count(distinct product) as num_sold,
string_agg(product, ',') within group (order by product) as products
from (select distinct sell_date, product from Activities) A
group by sell_date

-- =============================================================================
-- 175. Combine Two Tables

select Person.firstName, Person.lastName, Address.city, Address.state
from Person
left join Address
on Person.personId = Address.personId


select firstName, lastName, city, state
from Person
left join Address
on Person.personId = Address.personId

-- =============================================================================
-- 1148. Article Views I

select distinct author_id as id from Views
where author_id = viewer_id
order by author_id;

-- =============================================================================
-- 182. Duplicate Emails

select email as Email
from Person
group by email
having count(email) > 1;


select email as Email
from Person
group by email
having count(email) > 1;

-- =============================================================================
-- 577. Employee Bonus

select  Employee.name, Bonus.bonus
from Employee
left join Bonus
on Employee.empId = Bonus.empId
where bonus < 1000 or bonus is null

-- =============================================================================
-- 1527. Patients With a Condition

select * from Patients
where conditions like 'DIAB1%' or conditions like '% DIAB1%';

-- =============================================================================
-- 1393. Capital Gain/Loss

with new_stocks as (
    select stock_name, operation, operation_day, price, iif(operation = 'Buy', -price, price) as new_price
from Stocks
)

select stock_name, sum(new_price) as capital_gain_loss
from new_stocks
group by stock_name

-- =============================================================================
-- 620. Not Boring Movies

select * from Cinema
where description != 'boring'
and id % 2 != 0
order by rating desc;

-- =============================================================================
-- 1965. Employees With Missing Information

select ISNULL(Employees.employee_id, Salaries.employee_id) as employee_id
from Employees full join Salaries
on Employees.employee_id = Salaries.employee_id
where Employees.name is null or Salaries.salary is null
order by employee_id;


select employee_id
from Employees
where employee_id
not in(select employee_id from salaries)
union
select employee_id
from salaries
where employee_id
not in(select employee_id from Employees)

select employee_id from employees
union
select employee_id from salaries
except
select employee_id from employees
intersect
select employee_id from salaries

-- =============================================================================
-- 610. Triangle Judgement

select x, y, z, iif(x+y > z and y+z > x and x+z > y, 'Yes', 'No') as triangle
from Triangle;

select *, iif(x+y > z and y+z > x and x+z > y, 'Yes', 'No') as triangle
from Triangle;

select x, y, z, iif(x+y <= z or y+z <= x or x+z <= y, 'No', 'Yes') as triangle
from Triangle;

select x, y, z,
case
when x+y>z and x+z>y and y+z>x then 'Yes'
else 'No'
end as triangle
from Triangle

-- =============================================================================
-- 1327. List the Products Ordered in a Period

select Products.product_name, sum(Orders.unit) as unit
from Products right join Orders
on Products.product_id = Orders.product_id
where Orders.order_date >= '2020-02-01' and Orders.order_date <= '2020-02-29'
group by Orders.product_id
having unit >= 100;

select Products.product_name, sum(Orders.unit) as unit
from Products right join Orders
on Products.product_id = Orders.product_id
where Orders.order_date between '2020-02-01' and '2020-02-29'
group by Orders.product_id
having unit >= 100;

select Products.product_name, sum(Orders.unit) as unit
from Products right join Orders
on Products.product_id = Orders.product_id
where year(order_date) = 2020 and month(order_date) = 2
group by Orders.product_id
having unit >= 100;

-- =============================================================================
-- 584. Find Customer Referee

select name from  Customer
where referee_id != '2' or referee_id is null;

select name from Customer
where isnull(referee_id,0) !=2 ;

-- =============================================================================
-- 1729. Find Followers Count

select user_id, count(follower_id) as followers_count
from Followers
group by user_id
order by user_id;

select user_id, count(user_id) as followers_count
from Followers
group by user_id

-- =============================================================================
-- 511. Game Play Analysis I

select player_id, min(event_date) as first_login
from Activity
group by player_id;

-- =============================================================================
-- 1050. Actors and Directors Who Cooperated At Least Three Times

select actor_id, director_id from ActorDirector
group by actor_id, director_id
having count(actor_id) >= 3; --  При группировке по двум полям, каждая группа будет уникальной комбинацией actor_id и director_id. не может быть чтобы актер встретился с директором больше чем директор с актером. тк это одна и также встреча

select actor_id, director_id from ActorDirector
group by actor_id, director_id
having count(actor_id) >= 3 and count(director_id) >= 3;

select actor_id, director_id from ActorDirector
group by actor_id, director_id
having count(*) >= 3;



-- =============================================================================
-- 1661. Average Time of Process per Machine

with new_Activity as (
    select machine_id, process_id, activity_type,
    iif(activity_type = 'start', -timestamp, timestamp) as timestamp1
from Activity
)

select machine_id, round(avg(timestamp2), 3) as processing_time -- round - округление до N числа знаков после запятой
from (
    select machine_id, process_id, sum(timestamp1) as timestamp2
    from new_Activity
    group by machine_id, process_id
    ) summed_Activity
group by machine_id



-- =============================================================================
-- 595. Big Countries

select name, population, area from World
where area >= 3000000 or population >= 25000000;

-- =============================================================================
-- 1581. Customer Who Visited but Did Not Make Any Transactions

select customer_id, count(iif(transaction_id = null, 1, 0)) as count_no_trans
from Visits left join Transactions
on Visits.visit_id = Transactions.visit_id
where transaction_id is Null
group by customer_id;

select customer_id, count(Visits.visit_id) as count_no_trans
from Visits left join Transactions
on Visits.visit_id = Transactions.visit_id
where transaction_id is Null
group by customer_id;

select customer_id, count(1) as count_no_trans
from Visits left join Transactions
on Visits.visit_id = Transactions.visit_id
where transaction_id is Null
group by customer_id;

-- =============================================================================
--

with new_Weather as (
    select id, recordDate as recordDate2, temperature as temperature2
    from Weather
)

select Weather.id as Id
from Weather
join new_Weather
on Weather.recordDate = DATEADD(day, 1, new_Weather.recordDate2)
where new_Weather.temperature2 < Weather.temperature;


SELECT
    w1.id
FROM
    Weather w1
JOIN
    Weather w2
ON
    DATEDIFF(day, w2.recordDate, w1.recordDate) = 1
WHERE
    w1.temperature > w2.temperature;

-- =============================================================================
-- 1141. User Activity for the Past 30 Days I

select activity_date as day, count(distinct user_id) as active_users from Activity
where activity_date between '2019-06-28' and '2019-07-27'
and (activity_type != 'open_session' or activity_type != 'end_session') -- можно не использовать
group by activity_date

select activity_date as day, count(distinct user_id) as active_users from Activity
where activity_date <='2019-07-27' and activity_date > dateadd(day, -30, '2019-07-27')
and (activity_type != 'open_session' or activity_type != 'end_session') -- можно не использовать
group by activity_date

select activity_date as day, count(distinct user_id) as active_users from Activity
where  DATEDIFF(day,activity_date,'2019-07-27')<30 and DATEDIFF(day,activity_date,'2019-07-27')>=0
and (activity_type != 'open_session' or activity_type != 'end_session') -- можно не использовать
group by activity_date

select activity_date as day, count(distinct user_id) as active_users from Activity
where  DATEDIFF(day,activity_date,'2019-07-27')<30 and DATEDIFF(day,activity_date,'2019-07-27')>=0
group by activity_date

-- =============================================================================
-- 596. Classes More Than 5 Students

select class from Courses
group by class
having count(class) > 4

-- =============================================================================
-- 1075. Project Employees I

# MS SQL
select Project.project_id, round(avg(cast(Employee.experience_years as decimal)),2) as average_years -- cast() приведение к типу, decimal - нецелое число
from Project join Employee
on Project.employee_id = Employee.employee_id
group by Project.project_id

select Project.project_id, round(avg(Employee.experience_years*1.0),2) as average_years
from Project join Employee
on Project.employee_id = Employee.employee_id
group by Project.project_id

# MySQL
select Project.project_id, round(avg(Employee.experience_years), 2) as average_years
from Project join Employee
on Project.employee_id = Employee.employee_id
group by Project.project_id

-- =============================================================================
-- 619. Biggest Single Number

select isnull((select top 1 num from MyNumbers -- или  COALESCE()
group by num
having count(num) = 1
order by num desc), null)
as num

-- =============================================================================
-- 196. Delete Duplicate Emails

with New as (
    select min(id) as id, email from Person group by email having count(email) > 1
)

delete from Person
where email in (select email from New) and id not in (select id from New);

delete from Person
where id NOT IN (select min(id) from person
group by email)

delete p1
from Person p1, Person p2
where p1.Email = p2.Email
and p1.id>p2.Id;

-- =============================================================================
-- 1517. Find Users With Valid E-Mails

-- MySQL
select user_id, name, mail from Users
where mail regexp '^[a-zA-Z][a-zA-Z0-9_.-]*@leetcode[.]com$' -- MySQL

select user_id, name, mail from Users
where mail regexp '^[A-Za-z][A-Za-z0-9_.-]*@leetcode\\.com$'

-- MS SQL
select * from Users
where mail LIKE '[a-Z]%@leetcode.com' and substring(mail, 1, len(mail) - 13) not like '%[^0-9a-Z_.-]%'

select * from Users
WHERE mail LIKE '[a-z]%@leetcode.com'
AND user_id NOT IN (SELECT user_id FROM Users WHERE mail LIKE '%[^a-z0-9._-]%@leetcode.com')

-- =============================================================================
-- 1731. The Number of Employees Which Report to Each Employee

-- MySQL
select e2.employee_id, e2.name, count(e1.reports_to) as reports_count, round(avg(e1.age), 0) as average_age
from Employees e1 join Employees e2
on e2.employee_id = e1.reports_to
group by e2.employee_id, e2.name
having count(e1.reports_to) > 0
order by e2.employee_id;

-- MS SQL
select e2.employee_id, e2.name, count(e1.reports_to) as reports_count, round(avg(e1.age*1.00), 0) as average_age
from Employees e1 join Employees e2
on e2.employee_id = e1.reports_to
group by e2.employee_id, e2.name
having count(e1.reports_to) > 0 --здесь не обязательно из-за группировки
order by e2.employee_id


select e2.employee_id, e2.name, count(e1.reports_to) as reports_count, round(avg(e1.age*1.00), 0) as average_age
from Employees e1 join Employees e2
on e2.employee_id = e1.reports_to
group by e2.employee_id, e2.name
order by e2.employee_id

-- =============================================================================
-- 1978. Employees Whose Manager Left the Company

select employee_id
from Employees
where
salary < 30000
and manager_id not in (select employee_id from Employees)
order by employee_id;

-- =============================================================================
-- 1667. Fix Names in a Table

select user_id, upper(left(name, 1)) + lower(right(name, len(name)-1))  as name from Users
order by user_id;

-- =============================================================================
-- 1280. Students and Examinations

select Students.student_id, Students.student_name, Subjects.subject_name, count(Examinations.student_id) as attended_exams
from Students
cross join Subjects -- замена left join Subjects on 1 = 1 на CROSS JOIN Subjects - создает все возможные комбинации студентов и предметов.
left join Examinations
on Students.student_id = Examinations.student_id and Subjects.subject_name = Examinations.subject_name
group by Students.student_id, Students.student_name, Subjects.subject_name
order by Students.student_id, Subjects.subject_name;

-- =============================================================================
-- 570. Managers with at Least 5 Direct Reports

select name
from Employee e1
join (select managerId from Employee
group by managerId
having count(managerId) > 4) e2
on e1.id = e2.managerId

-- =============================================================================
-- 1045. Customers Who Bought All Products

select customer_id from Customer
group by customer_id
having count(distinct product_key) = (select count(product_key) from Product)

-- =============================================================================
-- 1070. Product Sales Analysis III

select s1.product_id, s2.first_year, quantity, price
from Sales s1 join (
    select product_id, min(year) as first_year from Sales
    group by product_id
    ) s2
on s1.product_id = s2.product_id and s1.year = s2.first_year

-- =============================================================================
-- 1789. Primary Department for Each Employee

-- MS SQL
select employee_id, department_id
from Employee
where primary_flag = 'Y'

union

select employee_id, max(department_id) -- макс для галочки, тк нужно из-за групп бай. тк у нас будет выборка по одному значению макс не влияет
from Employee
group by employee_id
having count(employee_id) = 1;

SELECT employee_id, department_id
FROM Employee
WHERE primary_flag = 'Y' OR employee_id IN (
    SELECT employee_id
    FROM Employee
    GROUP BY employee_id
    HAVING COUNT(primary_flag) = 1
)

--MySQL
select employee_id, department_id
from Employee
where primary_flag = 'Y'

union

select employee_id, department_id
from Employee
group by employee_id
having count(employee_id) = 1;
-- =============================================================================
-- 176. Second Highest Salary

select max(salary) as SecondHighestSalary
from Employee
where salary < (select max(salary) from Employee);

-- не обрабатывает дубли
select isnull((select top 1 e2.salary as SecondHighestSalary
from Employee e1 join Employee e2
on e1.id + 1 = e2.id
order by e1.salary), null)
as SecondHighestSalary;

-- =============================================================================
-- 1934. Confirmation Rate

select Signups.user_id,
round(iif(count(action) = 0, 0.0, sum(iif(action='confirmed', 1, 0)) / (count(action) * 1.0)), 2)
as confirmation_rate
from Signups
left join Confirmations
on Confirmations.user_id = Signups.user_id
group by Signups.user_id

-- =============================================================================
-- 1251. Average Selling Price

select Prices.product_id,
isnull(round(sum(Prices.price * UnitsSold.units * 1.0) / sum(UnitsSold.units),2), 0)
as average_price
from Prices left join UnitsSold
on Prices.product_id = UnitsSold.product_id
and UnitsSold.purchase_date between start_date and Prices.end_date
group by Prices.product_id

-- =============================================================================
-- 1633. Percentage of Users Attended a Contest

select contest_id,
round(count(user_id) * 100.0 / (select count(user_id) from Users), 2) as percentage
from Register
group by contest_id
order by percentage desc, contest_id

-- =============================================================================
--

select query_name,
round(avg(1.0 * rating / position), 2)
as quality,
round(sum(iif(rating<3, 100.0, 0)) / count(rating), 2)
as poor_query_percentage
from Queries
group by query_name
having query_name is not null;

-- =============================================================================
-- 1193. Monthly Transactions I

select
format(trans_date, 'yyyy-MM') as month,
country,
count(state) as trans_count,
sum(iif(state='approved', 1, 0)) as approved_count,
sum(amount) as trans_total_amount,
sum(iif(state='approved', amount, 0)) as approved_total_amount
from Transactions
group by format(trans_date, 'yyyy-MM'), country


-- =============================================================================
-- 1174. Immediate Food Delivery II

with New as (
    select
        distinct customer_id,
        min(order_date) as order_date,
        min(customer_pref_delivery_date) as customer_pref_delivery_date
    from Delivery
    group by customer_id
)

select
round(sum(iif(order_date=customer_pref_delivery_date, 100.0, 0)) / count(order_date), 2)
as immediate_percentage
from New



-- =============================================================================
-- 550. Game Play Analysis IV

with stats as (
        select player_id, min(event_date) as first_date
        from activity
        group by player_id
    ),
    logged_users as (
        select activity.player_id
        from activity
        join stats on activity.player_id = stats.player_id
        where datediff(day, first_date, event_date) = 1
    )

select
    round(count(logged_users.player_id) * 1.0 /
    (select count(distinct player_id) from activity), 2)
    as fraction
from logged_users



-- =============================================================================
-- 180. Consecutive Numbers

select distinct l1.num as ConsecutiveNums
from Logs l1, Logs l2, Logs l3
where l1.id = l2.id - 1 and l2.id = l3.id - 1 and l1.num = l2.num  and l2.num = l3.num

--не менее 3х раз ПОДРЯД

-- =============================================================================
-- 1164. Product Price at a Given Date

select
 product_id,
 isnull((
    select top 1 new_price
    from Products p2
    where p1.product_id = p2.product_id and change_date <= '2019-08-16'
    order by change_date desc
 ), 10) as price
from Products p1
group by product_id

-- =============================================================================
-- 585. Investments in 2016

select
round(sum(tiv_2016*1.0), 2) as tiv_2016
from Insurance
where tiv_2015 in
    (
    select tiv_2015
    from Insurance
    group by tiv_2015
    having count(tiv_2015) > 1
    )
and lat * 100 + lon in
    (
    select lat * 100 + lon as latlon
    from Insurance
    group by lat, lon
    having count(lat * 100 + lon) < 2
    );

-- =============================================================================
-- 602. Friend Requests II: Who Has the Most Friends

with New as(
select requester_id as id from RequestAccepted
union all
select accepter_id as id from RequestAccepted
)

select top 1 id, count(id) as num
from New
group by id
order by num desc



select top 1 id, count(id) as num
from (
select requester_id as id from RequestAccepted
union all
select accepter_id as id from RequestAccepted
) New
group by New.id
order by num desc


-- =============================================================================
-- 1341. Movie Rating

with top_user as (
    select top 1 name as results
    from (select u.user_id, u.name, m.title, mr.rating
        from MovieRating mr
        join Users u
            on mr.user_id = u.user_id
        join Movies m
            on mr.movie_id = m.movie_id) new1
    group by new1.user_id, new1.name
    order by count(new1.user_id) desc, new1.name
),
top_movie as(
    select top 1 title as results
    from (
        select m.title, avg(mr.rating*1.0) as rating
        from MovieRating mr
        join Movies m on mr.movie_id = m.movie_id
        where mr.created_at between '2020-02-01' and '2020-02-29'
        group by mr.movie_id, m.title
        ) new2
    order by new2.rating desc, new2.title
)

select results from top_user
union all
select results from top_movie

-- =============================================================================
-- 1204. Last Person to Fit in the Bus
-- MS SQL
select top 1 person_name
from (
    select
        person_id, person_name, weight, turn,
        sum(weight) over (
            order by turn
            rows between unbounded preceding and current row
        ) as weight_sum
    from Queue
) x
where x.weight_sum <= 1000
order by x.turn desc


-- MySql
select person_name from (
    select person_id, person_name, weight, turn,
    sum(weight) over w as weight_sum
    from Queue
    window w as (
        order by turn
        rows between unbounded preceding and current row
        )
) x
where x.weight_sum <= 1000
order by x.turn desc
limit 1

-- =============================================================================
-- 1327. List the Products Ordered in a Period

select Products.product_name, sum(Orders.unit) as unit
from Products right join Orders
on Products.product_id = Orders.product_id
where Orders.order_date >= '2020-02-01' and Orders.order_date <= '2020-02-29'
group by Orders.product_id
having unit >= 100;

-- =============================================================================
-- 1907. Count Salary Categories

select 'High Salary' as category, (select count(*) from Accounts where income > 50000) as accounts_count
union all
select 'Average Salary' as category, (select count(*) from Accounts where income >= 20000 and income <= 50000) as accounts_count
union all
select 'Low Salary' as category, (select count(*) from Accounts where income < 20000) as accounts_count


-- =============================================================================
-- 1321. Restaurant Growth
-- MySql

select
    c.visited_on,
    sum(c.amount) over w as amount,
    round(avg(c.amount)over w , 2) as average_amount
from (
    select visited_on, sum(amount) as amount
    from Customer
    group by visited_on
) c
window w as (
    rows between 6 preceding and current row
    )
order by c.visited_on
limit 1000000 offset 6

-- =============================================================================
-- 626. Exchange Seats

select
iif(
    id % 2 = 0,
    id - 1,
    iif(
        id+1 not in (select id from seat),
        id,
        id + 1
        )
    ) as id,
student
from Seat
order by id


-- =============================================================================
-- 185. Department Top Three Salaries (hard)
--mysql

select Department, Employee, Salary
    from (
    select
    dense_rank() over w as top,
    e.name as Employee, salary as Salary, departmentId, d.name as Department from Employee e
    join Department d
    on e.departmentId = d.id
    window w as (
    partition by departmentId
    order by salary desc
    )
) x
where top < 4

-- =============================================================================
--  178. Rank Scores

select
    score,
    dense_rank() over w as 'rank'
from Scores
window w as (
    order by score desc
    )


select
    score,
    dense_rank() over (order by score desc) as 'rank'
from Scores

-- =============================================================================
-- 181. Employees Earning More Than Their Managers
-- My SQL

select e1.name as Employee
from Employee e1
join Employee e2 on e1.managerId = e2.id
where e2.salary < e1.salary

-- =============================================================================
-- 1158. Market Analysis I

select
    u.user_id as buyer_id,
    u.join_date,
    count(o.order_date) as orders_in_2019
from Users u
left join orders o
on u.user_id = o.buyer_id and year(o.order_date) = '2019'
group by u.user_id

-- =============================================================================
-- 184. Department Highest Salary
--mySQL

select Department, Employee, Salary from (
    select
        dense_rank() over w as top,
        d.name as Department,
        e.name as Employee,
        salary as Salary
    from Employee e
    join Department d on e.departmentId = d.id
    window w as(
        partition by e.departmentId
        order by Salary desc
    )
) x
where top = 1;


select Department, Employee, Salary from (
    select
        dense_rank() over (
            partition by e.departmentId
            order by Salary desc
            ) as top,
        d.name as Department,
        e.name as Employee,
        salary as Salary
    from Employee e
    join Department d on e.departmentId = d.id
) x
where top = 1;

-- =============================================================================
-- 183. Customers Who Never Order

select name as Customers
from Customers
where id not in (select customerId from Orders)

-- =============================================================================
-- 1084. Sales Analysis III

-- MySQL
select product_id, product_name
from Product
where product_id in (
    select product_id from Sales
    where DATE_FORMAT(sale_date, '%Y-%m') BETWEEN '2019-01' AND '2019-03'
    )
    and product_id not in (
    select product_id from Sales
    where sale_date > '2019-03-31' or sale_date < '2019-01-01'
    )

-- =============================================================================
-- 607. Sales Person

with orders_red as (
    select SalesPerson.name
    from Orders
    left join Company on Company.com_id = Orders.com_id
    left join SalesPerson on SalesPerson.sales_id = Orders.sales_id
    where company.name = 'RED'
)

select name from SalesPerson
where name not in (select * from orders_red)


-- =============================================================================
-- 586. Customer Placing the Largest Number of Orders

select customer_number from
(select customer_number, count(order_number) from Orders
group by customer_number
order by count(order_number) desc) x
limit 1

-- =============================================================================
-- 1407. Top Travellers
-- MySQL
select name, coalesce(sum(distance), 0) as travelled_distance
from Users
left join Rides on Users.id = Rides.user_id
group by Users.id
order by travelled_distance desc, name


--MS SQL
select min(name) as name, isnull(sum(distance), 0) as travelled_distance
from Users
left join Rides on Users.id = Rides.user_id
group by Users.id
order by travelled_distance desc, name

-- =============================================================================
-- 3220. Odd and Even Transactions

-- MS SQL
select transaction_date,
sum(iif(amount % 2 != 0, amount, 0)) as odd_sum,
sum(iif(amount % 2 = 0, amount, 0)) as even_sum
from transactions
group by transaction_date
order by transaction_date;

-- MySQL
select transaction_date,
sum(if(amount % 2 != 0, amount, 0)) as odd_sum,
sum(if(amount % 2 = 0, amount, 0)) as even_sum
from transactions
group by transaction_date
order by transaction_date;


-- =============================================================================
-- 1873. Calculate Special Bonus

select employee_id,
if(employee_id % 2 != 0 and name not like 'M%', salary, 0) as bonus
from Employees
order by employee_id;

-- =============================================================================
--


-- =============================================================================
--


-- =============================================================================
--

