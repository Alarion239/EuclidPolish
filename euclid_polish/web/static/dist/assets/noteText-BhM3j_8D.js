function r(e){if(!e)return"—";const t=new Date(e);return Number.isNaN(t.getTime())?e:`${t.toISOString().slice(0,16).replace("T"," ")} UTC`}export{r as u};
