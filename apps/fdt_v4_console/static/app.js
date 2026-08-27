const $=id=>document.getElementById(id);
async function api(path,options={}){const response=await fetch(path,{cache:'no-store',...options});const body=await response.json();if(!response.ok)throw new Error(body.error||'요청 실패');return body}
function number(value,digits=6){const parsed=Number(value);return Number.isFinite(parsed)?parsed.toFixed(digits):'--'}
function eta(seconds){if(!Number.isFinite(Number(seconds)))return'--';let value=Math.max(0,Number(seconds));const days=Math.floor(value/86400);value%=86400;const hours=Math.floor(value/3600);value%=3600;const minutes=Math.floor(value/60);return days?`${days}일 ${hours}시간`:hours?`${hours}시간 ${minutes}분`:`${minutes}분`}
function notice(message){const box=$('notice');box.textContent=message;box.hidden=false;setTimeout(()=>box.hidden=true,5000)}
function render(state){$('loss').textContent=number(state.metrics?.loss);$('entropy').textContent=number(state.metrics?.entropy_normalized);$('remaining').textContent=eta(state.metrics?.remaining_seconds);$('status').textContent=state.status||'IDLE';const done=Number(state.metrics?.additional_tokens||0);const total=Number(state.metrics?.target_additional_tokens||0);$('progress').textContent=total?`${(done/1e6).toFixed(1)}M / ${(total/1e9).toFixed(2)}B`:'--';$('stop').disabled=!state.can_emergency_stop;$('resume').disabled=!state.can_resume}
async function poll(){try{render(await api('/api/status'))}catch(error){$('status').textContent='연결 대기'}}
$('stop').onclick=async()=>{if(!confirm('원자적 복구 체크포인트를 저장한 뒤 정지합니다.'))return;try{await api('/api/emergency-stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});notice('정지 요청을 전달했습니다. 체크포인트 저장을 기다립니다.')}catch(error){notice(error.message)}};
$('resume').onclick=async()=>{try{await api('/api/resume',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});notice('Luna 감독으로 학습을 재개했습니다.')}catch(error){notice(error.message)}};
poll();setInterval(poll,3000);
