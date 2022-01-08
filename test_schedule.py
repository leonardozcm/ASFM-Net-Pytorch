def getalpha(schedule, step):
    alpha = 0.0
    for (point, alpha_) in schedule:
        if step >= point:
            alpha = alpha_
        else:
            break
    return alpha


step_stage_0 = 0
step_stage_1 = 5e4
step_stage_2 = 7e4
step_stage_3 = 1e5
step_stage_4 = 2.5e5

step_stages = [step_stage_0, step_stage_1,
               step_stage_2, step_stage_3, step_stage_4]

schedule = [[1., 0., 0., 0., 0.],
            [0., 1., 1., 0., 0.],
            [0., 0.1, 0.5, 1.0, 0.9]]

schedule_new = []
for ls in schedule:
    ls_new = []
    for i, a in enumerate(ls):
        ls_new.append((step_stages[i], a))
    schedule_new.append(ls_new)

step_0 = 2e4
step_1 = 6e4
step_2 = 9e4
step_3 = 2e5
step_4 = 3e5

steps_test = [step_0, step_1, step_2, step_3, step_4]

for i in steps_test:
    print("="*10, " ", i, " ", "="*10)
    result = []
    for index in range(3):
        result.append(getalpha(schedule_new[index], i))
    print(result)
