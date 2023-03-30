var url =
  "https://api.coingecko.com/api/v3/coins/markets?vs_currency=Eur&order=market_cap_desc&per_page=100&page=1&sparkline=false&price_change_percentage=1m";
var cryptoList = [];
(function startWebsite() {
  $.ajax({
    url,
    method: "GET",
    dataType: "json",
    success: function (data) {
      cryptoList = data;
      print_tracking();
    },
  });
})();

function print_tracking() {

  let todoItems = [];
  let c = 1;

  cryptoList.forEach((item) => {
    let todoItem = document.createElement("div");
    todoItem.setAttribute("class", "crypto_row");
    todoItem.setAttribute("id", item.symbol);
    let appColor = item.price_change_percentage_24h < 0 ? "red" : "green";
    let decimals = item.current_price.toString().length;
    let priceToPrint = 0;
    let priceChange = 0;
    let ath = 0;
    //if the decimal is less than 0.1 print decimal - 4 digits
    if (item.ath < 0.1) {
        ath = item.ath.toLocaleString("it-IT", {
          style: "currency", 
          currency: "EUR", 
          minimumFractionDigits: decimals - 4,});
    } else {
        ath = item.ath.toLocaleString("it-IT", {
          style: "currency", 
          currency: "EUR",});
    }
    if (item.current_price < 0.1) {
        priceToPrint = item.current_price.toLocaleString("it-IT", {
          style: "currency", 
          currency: "EUR", 
          minimumFractionDigits: decimals - 4,});
        priceChange = item.price_change_24h.toLocaleString("it-IT", {
          style: "currency",
          currency: "EUR",
          minimumFractionDigits: decimals - 4,});
    } else {
        priceToPrint = item.current_price.toLocaleString("it-IT", {
          style: "currency",
          currency: "EUR",});
        priceChange = item.price_change_24h.toLocaleString("it-IT", {
          style: "currency",
          currency: "EUR",});
    }

    todoItem.innerHTML = `
                    <nId class="crypto_nid">${c}</nId>
                    <div class="crypto_img_box"><img class="crypto_img" src="${item.image}"></div>
                    <div class="crypto_symbol" ><b>${item.symbol.toUpperCase()}</b></div>
                    <div class="crypto_price">${priceToPrint}</div>
                    <div class="ath">${ath}</div>
                    <div class="perEur" style="color:${appColor}">${priceChange}</div>
                    <div class="crypto_price_perc" style="color:${appColor}"><b>
                    <span>${parseFloat(item.price_change_percentage_24h).toFixed(2)}</span>%</b>`;

    todoItem.addEventListener("click", function () {
      $("#main_title").html(`${item.name}`);
      $("#crypto_header").html("");
      $("#crypto_body").html(`
              <a class="go-back-arrow" href="/tracker"><i class='bx bx-arrow-back' style="font-size:25px"></i></a>
              <div class="cryptoElementNew">
                <div><img src="${item.image}"></div>
                <div><b>${item.symbol.toUpperCase()}</b></div>
                <div>${priceToPrint}</div>
                <div class="perEur" style="color:${appColor}">${priceChange}</div>
                <div style="color:${appColor}"><b><span>${parseFloat(item.price_change_percentage_24h).toFixed(2)}</span>%</b></div>
              </div>
              <div class="spanSpecialChild"><span>Daily Low: ${item.low_24h}€</span>
              <span>Daily High: ${item.high_24h}€ </span></div>
              <iframe src="https://bit2me.com/widget/chart/v1?currency=${item.symbol}&fiat=EUR" 
              style="display:block;width:100%;height:400px;margin:0 auto;" frameborder="0"></iframe>`);
    });

    todoItems.push(todoItem);
    c++;

  });

  document.querySelector("#crypto_body").replaceChildren(...todoItems);
  
}
