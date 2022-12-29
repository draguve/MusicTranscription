import url from 'node:url';
import * as fs from 'fs';
import pLimit from 'p-limit';

// Example Concurrency of 3 promise at once
const limit = pLimit(15);


function paramsToObject(entries) {
    const result = {}
    for(const [key, value] of entries) { // each 'entry' is a [key, value] tupple
        result[key] = value;
    }
    return result;
}

async function getItemsAtIndex(index){
    let test_url = "https://ignition4.customsforge.com/?draw=10&columns%5B0%5D%5Bdata%5D=addBtn&columns%5B0%5D%5Bsearchable%5D=false&columns%5B0%5D%5Borderable%5D=false&columns%5B1%5D%5Bdata%5D=artistName&columns%5B1%5D%5Bname%5D=artist.name&columns%5B2%5D%5Bdata%5D=titleName&columns%5B2%5D%5Bname%5D=title&columns%5B3%5D%5Bdata%5D=albumName&columns%5B3%5D%5Bname%5D=album&columns%5B4%5D%5Bdata%5D=year&columns%5B5%5D%5Bdata%5D=duration&columns%5B5%5D%5Borderable%5D=false&columns%5B6%5D%5Bdata%5D=tunings&columns%5B6%5D%5Bsearchable%5D=false&columns%5B6%5D%5Borderable%5D=false&columns%5B7%5D%5Bdata%5D=version&columns%5B7%5D%5Bsearchable%5D=false&columns%5B7%5D%5Borderable%5D=false&columns%5B8%5D%5Bdata%5D=memberName&columns%5B8%5D%5Bname%5D=author.name&columns%5B9%5D%5Bdata%5D=created_at&columns%5B9%5D%5Bsearchable%5D=false&columns%5B10%5D%5Bdata%5D=updated_at&columns%5B10%5D%5Bsearchable%5D=false&columns%5B11%5D%5Bdata%5D=downloads&columns%5B11%5D%5Bsearchable%5D=false&columns%5B12%5D%5Bdata%5D=parts&columns%5B12%5D%5Borderable%5D=false&columns%5B13%5D%5Bdata%5D=platforms&columns%5B14%5D%5Bdata%5D=file_pc_link&columns%5B14%5D%5Bsearchable%5D=false&columns%5B15%5D%5Bdata%5D=file_mac_link&columns%5B15%5D%5Bsearchable%5D=false&columns%5B16%5D%5Bdata%5D=artist.name&columns%5B17%5D%5Bdata%5D=title&columns%5B18%5D%5Bdata%5D=album&columns%5B19%5D%5Bdata%5D=author.name&columns%5B20%5D%5Bdata%5D=discussionID&order%5B0%5D%5Bcolumn%5D=10&order%5B0%5D%5Bdir%5D=desc&start=0&length=50&search%5Bvalue%5D=&table_filter_artist=&table_filter_album=&filter_title=&filter_album=&filter_preferred_platform=pc&filter_start_year=&filter_end_year=&filter_preferred=&filter_lovedCreators=&filter_official=&filter_rsplus=&filter_disable=&filter_notdisabled=&filter_hidden=&_=1672312450448"
    const myURL = url.parse(test_url);
    let newSearchParams = new URLSearchParams(myURL.search);
    newSearchParams.set("start",(index*50).toString());
    let data;
    data = await fetch(test_url.substring(0, test_url.indexOf("?")) + "?" + newSearchParams.toString(), {
        "headers": {
            "accept": "application/json, text/javascript, */*; q=0.01",
            "accept-language": "en-US,en;q=0.9",
            "sec-ch-ua": "\"Not?A_Brand\";v=\"8\", \"Chromium\";v=\"108\", \"Google Chrome\";v=\"108\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "x-csrf-token": "jUI4frG48nPUCGsFQryG7ftci84bTgLTLnbx2XWT",
            "x-requested-with": "XMLHttpRequest",
            "cookie": "ips4_ipsTimezone=Asia/Calcutta; ips4_device_key=52181e553eee96621b71686e578a9637; ips4_IPSSessionFront=a861fced7fd2861e22e05e4d2e7d233b; ips4_member_id=9837; ips4_login_key=bd0d26fb74b615746698c9de02fb241d; ips4_loggedIn=1672146966; ips4_forum_view=table; ips4_forum_list_view=list; ips4_hasJS=true; XSRF-TOKEN=eyJpdiI6ImYwSTNGR2FmdlRaOWNnTzFhdzdjdlE9PSIsInZhbHVlIjoiYkZnQkl6ZUp6MFdjYndaT0ZCUkw0NFE3VGR1YmhwS1BFWnpJU3IxRkdEUkZHRjM0VUlWRkNZQkI3ak93ck1RQUErNTJvZXlTRElGZUNYd05pVUljcDdzR1lOUEVWT1IwVGxwb2RhdVJSVlhOODdKMzVDTlAwM21LdmExMWswZnAiLCJtYWMiOiI0ZTI2OTdjNjQ4MmJkZjdhYTFkMWQ4ODdkMjRhMDhlYWQ1ZjFhMmUwMWEzOTUyN2M4Y2Q1MjBlM2QzZWVkMjU2IiwidGFnIjoiIn0%3D; ignition4_session=eyJpdiI6InJ3N1N2Sk5FcG1scmxvOU5waGtiK1E9PSIsInZhbHVlIjoiWC95dlJCbnJaekhxdHVobktoMVhBcXdCbUdTUTRBeU4xaGYxdkxMQllHV0oyV0h1YmVPYjJLU0lpUDNzbDlnaFZ1a1lnZTFIQmprUHBtVGY1NHNjK1ExM3QvR3d5dnluQ01OQUljNG9Ga3U3eFhVVTBsTFhvU0U3TlBOaklQQzYiLCJtYWMiOiJiODI3YzdiZTgwODU1NzcxYjVlOTU2NGMzMDczODM5OGRmOWUxOTVlZjYyM2I3YzM0MTU5OTA3MjVmZWE1YTkxIiwidGFnIjoiIn0%3D; remember_web_59ba36addc2b2f9401580f014c7f58ea4e30989d=eyJpdiI6IjBsQm9YWnlqQUs2L2xzekVUbmU3SFE9PSIsInZhbHVlIjoibXcyVUNLazVZc0w0aTFTZWxiRkRKN3diQjhZdkhoT280WXl3N3YyTldZcTFWajJsVjBBRUJaY1pybUt0RzZnZFNndkRTd1ltclI2RENaTnd3eXVweG5tcm9IaU92RWswNE9rU0k3OGtpUlE5ZGpXT2xVQ0p2L2tNZkFoTDJMSkJWSXVaTHVkRUpoYWNLc3BiWUxzOHJRPT0iLCJtYWMiOiIzMTE0OGIzZDEyMTk5NDY5ZmVlZDBhYzMwMGMwMzRjYmEzOTM5NDU2MGM3MjU2MjI3NDdiMGZkYTZlYzFkZWRmIiwidGFnIjoiIn0%3D",
            "Referer": "https://ignition4.customsforge.com/",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        },
        "body": null,
        "method": "GET"
    });
    data = await data.json();
    console.log(index);
    if("data" in data)
        return data["data"];
    return []
}

let all_data = []
var promises = [];
for(let i = 0; i < 1218; i++) {
    const promise = limit(() => getItemsAtIndex(i));
    promises.push(promise)
}
const datas = await Promise.all(promises);
for(let i=0;i<datas.length;i++){
    for(let j=0;j<=datas[i].length;j++){
        if(datas[i][j] !== undefined){
            all_data.push(datas[i][j])
        }
    }
}

fs.writeFile("links.json", JSON.stringify(all_data), function(err) {
    if (err) {
        console.log(err);
    }
});
