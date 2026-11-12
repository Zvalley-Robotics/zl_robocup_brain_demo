#!/usr/bin/env python3
"""分析 Brain 定位日志；轨迹是估计，不是外部真值。"""
import argparse
import collections
import math
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

N = r'([-+\d.eE]+)'
P = r'\('+','.join([N]*3)+r'\)'
TRACE = re.compile(r'\[BRAIN_TRACE\] t_ns=(\d+) odom='+P+r' field='+P+r' otf='+P+r' calib=(\d+)')
EVENT = re.compile(r'\[BRAIN_LOC_OK\] t_ns=(\d+) method=(\S+) pose='+P)
SEARCH = re.compile(r'\[(\d+\.\d+)\].*\[LOC_SEARCH\] mode=(\S+) code=(\d+) success=(\d+) markers=(\d+) residual='+N+r' ms='+N)
SCHED = re.compile(r'\[LOC_SCHED\] method=(\w+) skipped=(\d+) attempts=(\d+) success=(\d+)')

def parse(path):
    traces, events, search, sched = [], [], [], {}
    for line in path.read_text(errors='replace').splitlines():
        if m := TRACE.search(line):
            v=m.groups(); traces.append([int(v[0])*1e-9, *map(float,v[1:10]), int(v[10])])
        if m := EVENT.search(line):
            v=m.groups(); events.append((int(v[0])*1e-9,v[1],*map(float,v[2:])))
        if m := SEARCH.search(line):
            v=m.groups(); search.append((float(v[0]),v[1],int(v[2]),int(v[3]),int(v[4]),float(v[5]),float(v[6])))
        if m := SCHED.search(line):
            v=m.groups(); sched[v[0]]=tuple(map(int,v[1:]))
    return traces,events,search,sched

def main():
    ap=argparse.ArgumentParser(description='Brain 定位轨迹及调度分析')
    ap.add_argument('--log',required=True); ap.add_argument('--out',default='brain_loc_analysis.png')
    ap.add_argument('--field-length',type=float,default=14.0); ap.add_argument('--field-width',type=float,default=8.5)
    args=ap.parse_args(); traces,events,search,sched=parse(Path(args.log))
    fonts={f.name for f in font_manager.fontManager.ttflist}
    for name in ('Noto Sans CJK SC','Noto Sans CJK JP','WenQuanYi Zen Hei','SimHei'):
        if name in fonts: plt.rcParams['font.sans-serif']=[name]; break
    plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(2,2,figsize=(17,12)); ax=axs[0,0]
    L,W=args.field_length,args.field_width
    ax.add_patch(plt.Rectangle((-L/2,-W/2),L,W,facecolor='#d9f1de',edgecolor='#286338',linewidth=2)); ax.axvline(0,color='#286338'); ax.add_patch(plt.Circle((0,0),0.75,fill=False,color='#286338'))
    if traces:
        a=np.array(traces); idx=np.flatnonzero(a[:,10]==1)
        # 时间回退时断线，避免把不连续运行误连起来。
        field=a[:,4:6].copy(); field[np.r_[False,np.diff(a[:,0])<0]]=np.nan
        ax.plot(field[:,0],field[:,1],color='#df2525',label='Brain 场地定位',linewidth=1.5)
        if len(idx):
            k=idx[0]; yaw=a[k,9]; c,s=math.cos(yaw),math.sin(yaw)
            x=a[k:,1]*c-a[k:,2]*s+a[k,7]; y=a[k:,1]*s+a[k:,2]*c+a[k,8]
            broken=np.r_[False,np.diff(a[k:,0])<0]; x[broken]=np.nan;y[broken]=np.nan
            ax.plot(x,y,'--',color='#006ef0',label='本体里程计（首次标定固定变换）',alpha=.8)
        ax.scatter([a[0,4]],[a[0,5]],marker='x',c='black',label='首个采样')
    colors={'ENTERFIELD':'#cc0077','TRUSTDIR':'#f39221','1M':'#27ab4a','2T':'#16a5a4','PT':'#4d68d8','LT':'#8a43af','2X':'#c15e2d','Border':'#addb21'}
    for method in sorted({e[1] for e in events}):
        e=[x for x in events if x[1]==method]; ax.scatter([x[2] for x in e],[x[3] for x in e],s=30,marker='D' if method=='TRUSTDIR' else 'o',color=colors.get(method,'grey'),edgecolors='black',linewidths=.3,label=f'{method} 成功 ×{len(e)}',zorder=4)
    ax.set_title('场地位姿与定位成功事件（非真值）');ax.set_xlabel('场地 X（米）');ax.set_ylabel('场地 Y（米）');ax.set_aspect('equal');ax.grid(alpha=.25);ax.legend(fontsize=8,loc='best') if ax.get_legend_handles_labels()[0] else None
    ax=axs[0,1]
    if sched:
        names=list(sched); x=np.arange(len(names)); attempts=np.array([sched[n][1] for n in names]); success=np.array([sched[n][2] for n in names]); rates=np.divide(success,attempts,out=np.zeros(len(names)),where=attempts>0)*100
        ax.bar(x,rates,color='#3a9d75');ax.set_xticks(x,names,rotation=30);ax.set_ylabel('成功 / 执行（%）');
        for i,(n,su,at) in enumerate(zip(names,success,attempts)): ax.text(i,rates[i],f'{su}/{at}',ha='center',fontsize=8)
    else: ax.text(.5,.5,'没有调度汇总',ha='center',transform=ax.transAxes)
    ax.set_title('最后一条调度汇总（累计，非独立帧）'); ax.grid(axis='y',alpha=.3)
    ax=axs[1,0]; codes=collections.Counter(x[2] for x in search if not x[3]); labels={2:'残差超限',4:'标志不足',1:'采样失败',3:'迭代未收敛',5:'概率计算失败'}
    if codes: ax.bar([labels.get(k,f'代码 {k}') for k in codes],list(codes.values()),color='#f09942');ax.tick_params(axis='x',rotation=20)
    else: ax.text(.5,.5,'没有搜索失败记录',ha='center',transform=ax.transAxes)
    ax.set_ylabel('次数');ax.set_title('粒子搜索失败原因（仅记录的搜索）')
    ax=axs[1,1]
    if search:
        t0=search[0][0]; ax.scatter([x[0]-t0 for x in search],[x[6] for x in search],c=['#24a148' if x[3] else '#d33333' for x in search],s=20)
        ax.set_xlabel('距首次搜索日志的时间（秒）');ax.set_ylabel('搜索耗时（毫秒）');ax.grid(alpha=.3)
    else: ax.text(.5,.5,'无 LOC_SEARCH（需开启 diagnostics）',ha='center',transform=ax.transAxes)
    ax.set_title('粒子搜索耗时（绿：成功；红：失败）')
    fig.tight_layout();fig.savefig(args.out,dpi=140);plt.close(fig)
    print(f'TRACE={len(traces)} 成功事件={len(events)} 搜索={len(search)}；输出={args.out}')
    failures=collections.Counter()
    for line in Path(args.log).read_text(errors='replace').splitlines():
        match=re.search(r'/locate/(1m|2t|pt|lt|2x|border)/fail, Failed, ([^\x1b\n]+)', line)
        if match:
            reason=re.sub(r'[-+]?\d+(?:\.\d+)?', 'N', match.group(2))
            failures[(match.group(1),reason)]+=1
    for (method,reason),count in failures.most_common(15): print(f'几何失败 {method}: {count} 次；{reason}')
    if not traces: print('未发现 BRAIN_TRACE：不能绘制 Brain 轨迹；旧日志无法补出。')
    if not sched: print('未发现调度汇总，请启用 locator.diagnostics。')
if __name__=='__main__': main()
