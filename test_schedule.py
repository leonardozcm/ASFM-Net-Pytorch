def getalpha(schedule, step):
    alpha = 0.0
    for (point, alpha_) in schedule:
        if step>=point:
            alpha = alpha_
        else:
            break
    return alpha

step_stage_0 = 0
step_stage_1 = 5e4
step_stage_2 = 7e4
step_stage_3 = 1e5
step_stage_4 = 2.5e5

step_stages = [step_stage_0,step_stage_1,step_stage_2,step_stage_3,step_stage_4]

schedule=[[1.,0.,0.,0.],
          [0.,1.,1.,0.],
          [0.,0.1,0.5,1.0,0.9]]

schedule_new = []
for ls in schedule:
    ls_new = []
    for i, a in enumerate(ls):
        ls_new.append((step_stages[i], a))
    schedule_new.append(ls_new)

print(schedule_new)