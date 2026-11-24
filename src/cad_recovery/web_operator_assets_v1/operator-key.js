// The credential lives in the password field; never in URL or browser storage.
export function operatorRequest(request,key,origin=location.origin){
  const scope=/^\/v1\/jobs\/[0-9a-f-]{36}\/versions\/[1-9]\d*\/pages\/\d+\/tables(?:\/|$)/i;
  return (path,options={},asBlob=false)=>{
    const url=new URL(path,origin);
    if(url.origin!==origin)throw new Error('Недопустимый адрес сервиса.');
    const headers=new Headers(options.headers);
    headers.delete('X-Table-Accept-Key');
    const scoped=scope.test(url.pathname);
    const credential=scoped?key():'';
    if(credential)headers.set('X-Table-Accept-Key',credential);
    return request(path,{...options,headers:Object.fromEntries(headers.entries()),
      ...(scoped?{redirect:'error'}:{})},asBlob);
  };
}
